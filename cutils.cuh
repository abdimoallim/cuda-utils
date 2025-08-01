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
