#include "black_scholes.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

// Device function for the Black-Scholes formula
__device__ double black_scholes(double S, double K, double T, double r, double sigma, bool is_call) {
  double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
  double d2 = d1 - sigma * sqrt(T);
  double call_price = S * normcdf(d1) - K * exp(-r * T) * normcdf(d2);
  return is_call ? call_price : call_price - S + K * exp(-r * T); // Put price
}

// Probability density function of the standard normal distribution
__device__ double normpdf(double x) {
  return exp(-0.5 * x * x) / sqrt(2.0 * CUDART_PI);
}

// Bisection method as fallback
__device__ double bisection(double S, double K, double T, double r, double market_price, bool is_call, double low, double high, double tolerance, int max_iterations) {
  double mid;
  for (int i = 0; i < max_iterations; ++i) {
    mid = (low + high) / 2.0;
    double price = black_scholes(S, K, T, r, mid, is_call);
    if (fabs(price - market_price) < tolerance) break;
    if (price < market_price) low = mid;
    else high = mid;
  }
  return mid;
}

// Kernel function to find implied volatility
__global__ void find_implied_volatility_kernel(
  const double* __restrict__ option_prices,
  const double* __restrict__ stock_prices,
  const double* __restrict__ strikes,
  const double* __restrict__ times,
  const double* __restrict__ risk_free_rate,
  double tolerance,
  int max_iterations,
  double* __restrict__ volatilities,
  int n) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    double S = stock_prices[tid];
    double K = strikes[tid];
    double T = times[tid];
    double market_price = option_prices[tid];
    double rfr = risk_free_rate[tid];

    double sigma = 0.2; // Initial guess
    double low = 0.01;
    double high = 10;
    bool converged = false;

    for (int i = 0; i < max_iterations; ++i) {
      double price = black_scholes(S, K, T, rfr, sigma, true); // Assuming call option
      double vega = S * sqrt(T) * normpdf((log(S / K) + (rfr + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T)));

      double diff = market_price - price;
      if (fabs(diff) < tolerance) {
        converged = true;
        break;
      }

      double update = diff / vega;
      if (fabs(update) > 1.0) {
        update = (update > 0) ? 1.0 : -1.0; // Limit update step size
      }

      sigma += update;

      // Ensure sigma remains within bounds
      if (sigma < low) {
        sigma = low;
        break;
      } else if (sigma > high) {
        sigma = high;
        break;
      }
    }

    // If not converged, use bisection method
    if (!converged) {
      sigma = bisection(S, K, T, rfr, market_price, true, low, high, tolerance, max_iterations);
    }

    volatilities[tid] = sigma;
  }
}

void find_implied_volatility_cuda(
  const double* option_prices,
  const double* stock_prices,
  const double* strikes,
  const double* times,
  const double* risk_free_rate,
  double tolerance,
  int max_iterations,
  double* volatilities,
  int n) {

  double *d_option_prices, *d_stock_prices, *d_strikes, *d_times, *d_risk_free_rates, *d_volatilities;
  cudaMalloc(&d_option_prices, n * sizeof(double));
  cudaMalloc(&d_stock_prices, n * sizeof(double));
  cudaMalloc(&d_strikes, n * sizeof(double));
  cudaMalloc(&d_times, n * sizeof(double));
  cudaMalloc(&d_volatilities, n * sizeof(double));
  cudaMalloc(&d_risk_free_rates, n * sizeof(double));

  cudaMemcpy(d_option_prices, option_prices, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_stock_prices, stock_prices, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_strikes, strikes, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_times, times, n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_risk_free_rates, risk_free_rate, n * sizeof(double), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  find_implied_volatility_kernel<<<numBlocks, blockSize>>>(
    d_option_prices, d_stock_prices, d_strikes, d_times,
    d_risk_free_rates, tolerance, max_iterations, d_volatilities, n);

  cudaMemcpy(volatilities, d_volatilities, n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_option_prices);
  cudaFree(d_stock_prices);
  cudaFree(d_strikes);
  cudaFree(d_times);
  cudaFree(d_risk_free_rates);
  cudaFree(d_volatilities);
}