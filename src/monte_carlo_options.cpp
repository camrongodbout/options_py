//
// Created by camrongodbout on 7/1/24.
//

#include "monte_carlo_options.h"
#include "monte_carlo.cuh"

// MonteCarlo class implementation
MonteCarloOptions::MonteCarloOptions() {
  // Constructor code if needed
}

MonteCarloOptions::~MonteCarloOptions() {
  // Destructor code if needed
}

float MonteCarloOptions::calculate_option_price(float s, float k, float t, float r, float sigma, bool is_call, int n_paths, int n_steps) {
  float dt = t / n_steps;
  size_t paths_size = n_paths * (n_steps + 1) * sizeof(float);
  size_t payoffs_size = n_paths * sizeof(float);

  float *d_paths, *d_payoffs;
  cudaMalloc((void**)&d_paths, paths_size);
  cudaMalloc((void**)&d_payoffs, payoffs_size);

  simulate_gbm_paths(d_paths, n_paths, n_steps, s, t, r, sigma, dt);
  calculate_payoffs(d_paths, d_payoffs, n_paths, n_steps, k, r, t, is_call);

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