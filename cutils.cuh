#include <cuda_runtime.h>
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
