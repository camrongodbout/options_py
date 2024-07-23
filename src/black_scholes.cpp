#include "black_scholes.h"
#include "black_scholes.cuh"
#include <vector>

std::vector<double> find_implied_volatility(
  const std::vector<double>& option_prices,
  const std::vector<double>& stock_prices,
  const std::vector<double>& strikes,
  const std::vector<double>& times,
  const std::vector<double>& risk_free_rate,
  const std::vector<bool>& isCalls,
  double tolerance,
  int max_iterations) {

  int n = option_prices.size();
  std::vector<double> volatilities(n);

  // Convert std::vector<bool> to std::vector<char>
  std::vector<char> isCalls_char(isCalls.begin(), isCalls.end());

  find_implied_volatility_cuda(
    option_prices.data(),
    stock_prices.data(),
    strikes.data(),
    times.data(),
    risk_free_rate.data(),
    isCalls_char.data(),
    tolerance,
    max_iterations,
    volatilities.data(),
    n);

  return volatilities;
}