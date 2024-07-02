#include "black_scholes.h"
#include "black_scholes.cuh"
#include <vector>

std::vector<double> find_implied_volatility(
  const std::vector<double>& option_prices,
  const std::vector<double>& stock_prices,
  const std::vector<double>& strikes,
  const std::vector<double>& times,
  double risk_free_rate,
  double tolerance,
  int max_iterations) {

  int n = option_prices.size();
  std::vector<double> volatilities(n);

  find_implied_volatility_cuda(
    option_prices.data(),
    stock_prices.data(),
    strikes.data(),
    times.data(),
    risk_free_rate,
    tolerance,
    max_iterations,
    volatilities.data(),
    n);

  return volatilities;
}