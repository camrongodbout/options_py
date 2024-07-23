//
// Created by camrongodbout on 7/1/24.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../src/monte_carlo_options.h"
#include "black_scholes.h"

float calculateImpliedVolatility(float market_price, float s, float k, float t, float r, bool is_call, float tol, int max_iterations, int n_paths, int n_steps) {
  MonteCarloOptions mc;
  float sigma_low = 0.001f;
  float sigma_high = 10.0f;

  for (int i = 0; i < max_iterations; ++i) {
    float sigma = (sigma_low + sigma_high) / 2.0f;
    float price_estimate = mc.calculate_option_price(s, k, t, r, sigma, is_call, n_paths, n_steps);

    if (fabs(price_estimate - market_price) < tol) {
      return sigma;
    } else if (price_estimate < market_price) {
      sigma_low = sigma;
    } else {
      sigma_high = sigma;
    }
  }

  return (sigma_low + sigma_high) / 2.0f;
}

namespace py = pybind11;

PYBIND11_MODULE(options_py, m) {
  m.doc() = "Python bindings for custom C++ library";

  m.def("implied_vol_monte_carlo", &calculateImpliedVolatility, "Calculate implied volatility using Monte Carlo method",
        pybind11::arg("market_price"), pybind11::arg("s"), pybind11::arg("k"), pybind11::arg("t"), pybind11::arg("r"),
        pybind11::arg("is_call"), pybind11::arg("tol"), pybind11::arg("max_iterations"), pybind11::arg("n_paths"), pybind11::arg("n_steps"));

  m.def("calculateImpliedVolatilityBlackScholes", &find_implied_volatility, "Find implied volatility using Black-Scholes model",
         py::arg("option_prices"), py::arg("stock_prices"), py::arg("strikes"), py::arg("times"),
         py::arg("risk_free_rate"), py::arg("isCalls"), py::arg("tolerance"), py::arg("max_iterations"));
}