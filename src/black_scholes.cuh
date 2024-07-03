#ifndef BLACK_SCHOLES_CUH
#define BLACK_SCHOLES_CUH

void find_implied_volatility_cuda(
  const double* option_prices,
  const double* stock_prices,
  const double* strikes,
  const double* times,
  const double* risk_free_rate,
  double tolerance,
  int max_iterations,
  double* volatilities,
  int n);

#endif // BLACK_SCHOLES_CUH