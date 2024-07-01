#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "monte_carlo.cuh"

__global__ void simulate_gbm_paths_kernel(float *d_paths, int n_paths, int n_steps, float s, float t, float r, float sigma, float dt) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_paths) return;

  curandState state;
  curand_init((unsigned long long) clock() + idx, 0, 0, &state);

  d_paths[idx * (n_steps + 1)] = s;
  for (int step = 1; step <= n_steps; ++step) {
    float z = curand_normal(&state);
    d_paths[idx * (n_steps + 1) + step] = d_paths[idx * (n_steps + 1) + step - 1] * expf((r - 0.5 * sigma * sigma) * dt + sigma * sqrtf(dt) * z);
  }
}

__global__ void calculate_payoffs_kernel(float *d_paths, float *d_payoffs, int n_paths, int n_steps, float k, float r, float t, bool is_call) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n_paths) return;

  float payoff = is_call ? fmaxf(d_paths[idx * (n_steps + 1) + n_steps] - k, 0.0f) : fmaxf(k - d_paths[idx * (n_steps + 1) + n_steps], 0.0f);
  d_payoffs[idx] = expf(-r * t) * payoff;
}

void simulate_gbm_paths(float *d_paths, int n_paths, int n_steps, float s, float t, float r, float sigma, float dt) {
  int threads = 256;
  int blocks = (n_paths + threads - 1) / threads;
  simulate_gbm_paths_kernel<<<blocks, threads>>>(d_paths, n_paths, n_steps, s, t, r, sigma, dt);
}

void calculate_payoffs(float *d_paths, float *d_payoffs, int n_paths, int n_steps, float k, float r, float t, bool is_call) {
  int threads = 256;
  int blocks = (n_paths + threads - 1) / threads;
  calculate_payoffs_kernel<<<blocks, threads>>>(d_paths, d_payoffs, n_paths, n_steps, k, r, t, is_call);
}