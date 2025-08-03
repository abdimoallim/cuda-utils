### cuda-utils

Collection of utilities for CUDA programming.

Tested on an NVIDIA Tesla T4 cloud GPU with the following configurations.

- **Compute Capability**: 7.5
- **Max Threads per Block**: 1024
- **Max Grid Size**: 2147483647 x 65535 x 65535
- **Shared Memory per Block**: 48 KB
- **Total Global Memory**: 14.56 GB
- **Memory Clock Rate**: 5.00 GHz
- **Memory Bus Width**: 256 bits
- **Warp Size**: 32

Refer to the [source](/cutils.cuh).

### API Reference

#### Error handling

- `CUDA_CHECK(call)` - Wraps CUDA runtime API calls with automatic error checking. Prints error and exits on failure.
- `CUDA_CHECK_KERNEL()` - Checks kernel launch and execution errors. Includes device synchronization.

#### Thread management

- `__device__ int get_global_id()` - Returns global thread index across entire grid.
- `__device__ int get_global_size()` - Returns total number of threads in grid.
- `__device__ bool is_thread_valid(int n)` - Bounds checking for thread-to-data mapping.
- `__device__ int get_warp_id()` - Returns warp index within current block.
- `__device__ int get_lane_id()` - Returns thread index within warp (0-31).
- `__device__ bool is_warp_leader()` - True if thread is first in warp.

#### Warp reductions

- `__device__ float warp_reduce_sum(float val)` - Shuffle-based warp sum reduction.
- `__device__ float warp_reduce_max(float val)` - Shuffle-based warp max reduction.
- `__device__ float warp_reduce_min(float val)` - Shuffle-based warp min reduction.
- `__device__ int warp_reduce_sum(int val)` - Integer version of warp sum.
- `__device__ int warp_reduce_max(int val)` - Integer version of warp max.
- `__device__ int warp_reduce_min(int val)` - Integer version of warp min.

#### Block reductions

- `template<typename T> __device__ T block_reduce_sum(T val)` - Two-stage block reduction using shared memory. Requires synchronization.
- `template<typename T> __device__ T block_reduce_max(T val)` - Block max reduction. Result valid only in thread 0.

#### Memory operations

- `template<typename T> __device__ void coalesced_load(T* shared_mem, const T* global_mem, int n)` - Coalesced load from global to shared memory using all block threads.

- `template<typename T> __device__ void coalesced_store(T* global_mem, const T* shared_mem, int n)` - Coalesced store from shared to global memory.

#### Atomic operations

- `__device__ float atomic_add_float(float* address, float val)` - Atomic add for single-precision float.
- `__device__ double atomic_add_double(double* address, double val)` - Atomic add for double-precision using CAS loop.
- `__device__ float atomic_max_float(float* address, float val)` - Atomic max for float using CAS loop.
- `__device__ float atomic_min_float(float* address, float val)` - Atomic min for float using CAS loop.

#### Vector operations

- `__device__ float3 make_float3_uniform(float val)` - Creates float3 with all components equal to val.
- `__device__ float3 operator+(const float3& a, const float3& b)` - Vector addition.
- `__device__ float3 operator-(const float3& a, const float3& b)` - Vector subtraction.
- `__device__ float3 operator*(const float3& a, float s)` - Vector-scalar multiplication.
- `__device__ float dot(const float3& a, const float3& b)` - Dot product.
- `__device__ float length(const float3& v)` - Vector magnitude.
- `__device__ float3 normalize(const float3& v)` - Unit vector using fast reciprocal square root.

#### Utility functions

- `template<typename T> __device__ void swap(T& a, T& b)` - Generic swap function.
- `template<typename T> __device__ T lerp(T a, T b, float t)` - Linear interpolation.
- `__device__ float smoothstep(float e0, float e1, float x)` - Hermite interpolation for smooth transitions.
- `__device__ bool approx_equal(float a, float b, float epsilon = 1e-6f)` - Floating-point equality comparison with tolerance.
- `__device__ int next_power_of_two(int x)` - Returns smallest power of 2 >= x.
- `__device__ bool is_power_of_two(int x)` - Tests if x is power of 2.

#### Parallel algorithms

- `template<int BLOCK_SIZE> __device__ void block_scan_exclusive(float* data)` - In-place exclusive prefix sum. Data must be in shared memory.
- `__device__ void grid_stride_loop(int n, void (*func)(int))` - Grid-striding loop pattern for large arrays.

#### Launch configuration

- `dim3 get_grid_size(int n, int block_size)` - Computes grid size respecting conventional hardware limits.
- `dim3 get_launch_params(int n, int& block_size)` - Determines optimal block size and grid dimensions.

#### Memory management

- `template<typename T> T* cuda_malloc(size_t n)` - Type-safe device memory allocation.
- `template<typename T> void cuda_free(T* ptr)` - Safe device memory deallocation.
- `template<typename T> void cuda_memcpy_h2d(T* dst, const T* src, size_t n)` - Host to device memory copy.
- `template<typename T> void cuda_memcpy_d2h(T* dst, const T* src, size_t n)` - Device to host memory copy.
- `template<typename T> void cuda_memset(T* ptr, int value, size_t n)` - Device memory initialization.

#### RAII memory management

- `cuda_ptr<T>` - RAII wrapper for device memory with move semantics. Non-copyable.
  **Methods:**

  - `cuda_ptr(size_t n)`: Allocate `n` elements
  - `T* get()`: Raw pointer access
  - `void reset(size_t n)`: Reallocate with new size

#### Device information

- `void print_cuda_device_info()` - Prints comprehensive device properties and capabilities.
- `void cuda_device_sync()` - Explicit device synchronization.

#### Constants

- `WARP_SIZE` (`32`) - Threads per warp.
- `MAX_THREADS_PER_BLOCK` (`1024`) - Maximum threads per block.
- `MAX_GRID_SIZE` (`65535`) - Maximum grid dimension.

#### Utility macros

- `DIV_CEIL(a, b)` - Integer ceiling division.
- `MIN(a, b)`, `MAX(a, b)` - Min/max without function call overhead.
- `CLAMP(x, min_val, max_val)` - Clamp value to range.

### License

Apache License 2.0.
