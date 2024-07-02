//
// Created by camrongodbout on 7/1/24.
//

#ifndef CUDA_OPTIONS_MONTE_CARLO_OPTIONS_H
#define CUDA_OPTIONS_MONTE_CARLO_OPTIONS_H

#include "monte_carlo_options.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath> // For expf and sqrtf
#include <iostream>



class MonteCarloOptions
{
public:
  MonteCarloOptions();
  ~MonteCarloOptions();

  float calculate_option_price(float s, float k, float t, float r, float sigma, bool is_call, int n_paths, int n_steps);

private:
  curandState *d_state; // curand states on device
  int n_paths_;
  unsigned long long seed_;
};


#endif //CUDA_OPTIONS_MONTE_CARLO_OPTIONS_H
