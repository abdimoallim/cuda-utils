#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

#define CUDA_CHECK_KERNEL() \
  do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA kernel error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
    CUDA_CHECK(cudaDeviceSynchronize()); \
  } while(0)

#define WARP_SIZE (1<<5)
#define MAX_THREADS_PER_BLOCK (1<<10)
#define MAX_GRID_SIZE ((1<<16)-1)

#define DIV_CEIL(a, b) (((a) + (b) - 1) / (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, min_val, max_val) (MIN(MAX(x, min_val), max_val))

template<typename T>
__device__ __forceinline__ T lerp(T a, T b, float t) {
  return a + t * (b - a);
}

__device__ __forceinline__ float smoothstep(float e0, float e1, float x) {
  float t = CLAMP((x - e0) / (e1 - e0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

template<typename T>
__device__ __forceinline__ void swap(T& a, T& b) {
  T temp = a;
  a = b;
  b = temp;
}

__device__ __forceinline__ bool approx_equal(float a, float b, float epsilon = 1e-6f) {
  return fabsf(a - b) < epsilon;
}

__device__ __forceinline__ int next_power_of_two(int x) {
  return 1 << (32 - __clz(x - 1));
}

__device__ __forceinline__ bool is_power_of_two(int x) {
  return x > 0 && (x & (x - 1)) == 0;
}

//////////////////////////////////////////////////
//////////   global thread primitives   //////////
//////////////////////////////////////////////////

__device__ __forceinline__ int get_global_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int get_global_size() {
  return gridDim.x * blockDim.x;
}

__device__ __forceinline__ bool is_thread_valid(int n) {
  return get_global_id() < n;
}

__device__ __forceinline__ void grid_stride_loop(int n, void (*func)(int)) {
  int idx = get_global_id();
  int stride = get_global_size();

  for (int i = idx; i < n; i += stride) {
    func(i);
  }
}

/////////////////////////////////////////
///////////   warp-level ops   //////////
/////////////////////////////////////////

__device__ __forceinline__ int get_warp_id() {
  return threadIdx.x / WARP_SIZE;
}

__device__ __forceinline__ int get_lane_id() {
  return threadIdx.x & (WARP_SIZE - 1);
}

__device__ __forceinline__ bool is_warp_leader() {
  return get_lane_id() == 0;
}

/////////////////////////////////////////////////////////
//////////   warp-level reduction primitives   //////////
/////////////////////////////////////////////////////////

// @todo: rewrite as explicit template instantiation with int & float

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }

  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }

  return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fminf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }

  return val;
}

__device__ __forceinline__ int warp_reduce_sum(int val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }

  return val;
}

__device__ __forceinline__ int warp_reduce_max(int val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }

  return val;
}

__device__ __forceinline__ int warp_reduce_min(int val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }

  return val;
}

/////////////////////////////////////////////////////////
//////////   block-level reduction primitives   /////////
/////////////////////////////////////////////////////////

template<typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
  __shared__ T warp_sums[MAX_THREADS_PER_BLOCK / WARP_SIZE];

  int wid = get_warp_id();
  int lid = get_lane_id();

  val = warp_reduce_sum(val);

  if (lid == 0) {
    warp_sums[wid] = val;
  }

  __syncthreads();

  if (wid == 0) {
    val = (lid < blockDim.x / WARP_SIZE) ? warp_sums[lid] : T(0);
    val = warp_reduce_sum(val);
  }

  return val;
}

template<typename T>
__device__ __forceinline__ T block_reduce_max(T val) {
  __shared__ T warp_maxs[MAX_THREADS_PER_BLOCK / WARP_SIZE];

  int wid = get_warp_id();
  int lid = get_lane_id();

  val = warp_reduce_max(val);

  if (lid == 0) {
    warp_maxs[wid] = val;
  }

  __syncthreads();

  if (wid == 0) {
    val = (lid < blockDim.x / WARP_SIZE) ? warp_maxs[lid] : val;
    val = warp_reduce_max(val);
  }

  return val;
}

template<int BLOCK_SIZE>
__device__ void block_scan_exclusive(float* data) {
  int tid = threadIdx.x;

  for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    float tmp = 0;

    if (tid >= stride) {
      tmp = data[tid - stride];
    }

    __syncthreads();

    if (tid >= stride) {
      data[tid] += tmp;
    }

    __syncthreads();
  }

  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      int left = 2 * tid + stride;
      int right = left + stride;
      if (right < BLOCK_SIZE) {
        data[right] += data[left];
      }
    }

    __syncthreads();
  }
}

/////////////////////////////////////////////////////////
//////////   coalesced/cooperative load/store   /////////
/////////////////////////////////////////////////////////

template<typename T>
__device__ __forceinline__ void coalesce_load(
  T* shared_mem, const T* global_mem, int n) {
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  for (int i = tid; i < n; i += block_size) {
    shared_mem[i] = global_mem[i];
  }

  __syncthreads();
}

template<typename T>
__device__ __forceinline__ void coalesce_store(
  T* global_mem,
  const T* shared_mem,
  int n
) {
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  __syncthreads();

  for (int i = tid; i < n; i += block_size) {
    global_mem[i] = shared_mem[i];
  }
}

////////////////////////////////////////////
//////////   extended atomic ops   /////////
////////////////////////////////////////////

__device__ __forceinline__ float atomic_add_float(float* address, float val) {
  return atomicAdd(address, val);
}

__device__ __forceinline__ double atomic_add_double(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
}

__device__ __forceinline__ float atomic_max_float(float* address, float val) {
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);

  return __int_as_float(old);
}

__device__ __forceinline__ float atomic_min_float(float* address, float val) {
  int* address_as_int = (int*)address;
  int old = *address_as_int, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed,
                    __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);

  return __int_as_float(old);
}

///////////////////////////////////
//////////   float3 ops   /////////
///////////////////////////////////

__device__ __forceinline__ float3 make_float3_uniform(float val) {
  return make_float3(val, val, val);
}

__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3& a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator*(float s, const float3& a) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float length(const float3& v) {
  return sqrtf(dot(v, v));
}

__device__ __forceinline__ float3 normalize(const float3& v) {
  float inv_len = __frsqrt_rn(dot(v, v));
  return v * inv_len;
}

//////////////////////////////////////
//////////   launch config   /////////
//////////////////////////////////////

inline dim3 get_grid_size(int n, int block_size) {
  int grid_size = DIV_CEIL(n, block_size);

  if (grid_size > MAX_GRID_SIZE) {
      grid_size = MAX_GRID_SIZE;
  }

  return dim3(grid_size);
}

inline dim3 get_launch_params(int n, int& block_size) {
  block_size = 256;

  if (n < 256) {
    block_size = next_power_of_two(n);
    block_size = MAX(block_size, 32);
  }

  return get_grid_size(n, block_size);
}

/*misc*/

inline void cuda_device_sync() {
  CUDA_CHECK(cudaDeviceSynchronize());
}

inline void print_cuda_device_info() {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  printf("Device %d: %s\n", device, prop.name);
  printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("  Max Grid Size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("  Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
  printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
  printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
  printf("  Warp Size: %d\n", prop.warpSize);
}
