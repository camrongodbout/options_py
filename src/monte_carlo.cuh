#ifndef MONTE_CARLO_CUH
#define MONTE_CARLO_CUH

void init_curand_states(curandState *state, int n_paths, unsigned long long seed);
void monte_carlo_simulation(float *d_paths, float *d_payoffs, curandState *state, int n_paths, int n_steps, float s, float t, float r, float sigma, float dt, float k, bool is_call);


#endif // MONTE_CARLO_CUH