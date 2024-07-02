//
// Created by camrongodbout on 7/1/24.
//

#include "monte_carlo_options.h"
#include "monte_carlo.cuh"

// MonteCarlo class implementation
MonteCarloOptions::MonteCarloOptions() : d_state(nullptr), n_paths_(0), seed_(1234ULL) {
  // Constructor code if needed
}

MonteCarloOptions::~MonteCarloOptions() {
  // Destructor code if needed
  if (d_state) {
    cudaFree(d_state);
  }
}

float MonteCarloOptions::calculate_option_price(float s, float k, float t, float r, float sigma, bool is_call, int n_paths, int n_steps) {
  n_paths_ = n_paths;

  if (d_state == nullptr) {
    cudaMalloc((void**)&d_state, n_paths * sizeof(curandState));
    init_curand_states(d_state, n_paths, seed_);
  }

  float dt = t / n_steps;
  size_t paths_size = n_paths * (n_steps + 1) * sizeof(float);
  size_t payoffs_size = n_paths * sizeof(float);

  float *d_paths, *d_payoffs;
  cudaMalloc((void**)&d_paths, paths_size);
  cudaMalloc((void**)&d_payoffs, payoffs_size);

  monte_carlo_simulation(d_paths, d_payoffs, d_state, n_paths, n_steps, s, t, r, sigma, dt, k, is_call);

  float *h_payoffs = (float*)malloc(payoffs_size);
  cudaMemcpy(h_payoffs, d_payoffs, payoffs_size, cudaMemcpyDeviceToHost);

  float sum_payoffs = 0.0f;
  for (int i = 0; i < n_paths; ++i) {
    sum_payoffs += h_payoffs[i];
  }

  cudaFree(d_paths);
  cudaFree(d_payoffs);
  free(h_payoffs);

  return sum_payoffs / n_paths;
}