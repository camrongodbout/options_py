#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "monte_carlo.cuh"

// Kernel to initialize curand states
__global__ void init_curand_kernel(curandState *state, unsigned long long seed, int n_paths) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n_paths) {
    curand_init(seed, idx, 0, &state[idx]);
  }
}

__global__ void monte_carlo_kernel(float *d_paths, float *d_payoffs, curandState *state, int n_paths, int n_steps, float s, float t, float r, float sigma, float dt, float k, bool is_call) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_paths) return;

  extern __shared__ float shared_paths[];
  curandState localState = state[idx];

  d_paths[idx * (n_steps + 1)] = s;
  shared_paths[threadIdx.x] = s;

  for (int step = 1; step <= n_steps; ++step) {
    float z = curand_normal(&localState);
    shared_paths[threadIdx.x] = shared_paths[threadIdx.x] * expf((r - 0.5 * sigma * sigma) * dt + sigma * sqrtf(dt) * z);
    __syncthreads();
    d_paths[idx * (n_steps + 1) + step] = shared_paths[threadIdx.x];
  }

  float final_price = shared_paths[threadIdx.x];
  float payoff = is_call ? fmaxf(final_price - k, 0.0f) : fmaxf(k - final_price, 0.0f);
  d_payoffs[idx] = expf(-r * t) * payoff;

  state[idx] = localState;
}

void monte_carlo_simulation(float *d_paths, float *d_payoffs, curandState *state, int n_paths, int n_steps, float s, float t, float r, float sigma, float dt, float k, bool is_call) {
  int threads = 256;
  int blocks = (n_paths + threads - 1) / threads;
  size_t shared_memory_size = threads * sizeof(float);
  monte_carlo_kernel<<<blocks, threads, shared_memory_size>>>(d_paths, d_payoffs, state, n_paths, n_steps, s, t, r, sigma, dt, k, is_call);
}

void init_curand_states(curandState *state, int n_paths, unsigned long long seed) {
  int threads = 256;
  int blocks = (n_paths + threads - 1) / threads;
  init_curand_kernel<<<blocks, threads>>>(state, seed, n_paths);
}