#ifndef MONTE_CARLO_CUH
#define MONTE_CARLO_CUH

void simulate_gbm_paths(float *d_paths, int n_paths, int n_steps, float s, float t, float r, float sigma, float dt);
void calculate_payoffs(float *d_paths, float *d_payoffs, int n_paths, int n_steps, float k, float r, float t, bool is_call);

#endif // MONTE_CARLO_CUH