// Simulate a mesenchyme-to-epithelium transition
#include <assert.h>

#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"
#include "../lib/epithelium.cuh"


const float R_MAX = 1;
const float R_MIN = 0.6;
const int N_CELLS = 250;
const int N_TIME_STEPS = 100;
const float DELTA_T = 0.1;

__device__ __managed__ Solution<pocell, N_CELLS, LatticeSolver> X;


// Cubic potential plus k*(n_i . r_ij/r)^2/2 for all r_ij <= R_MAX
__device__ pocell epithelium(pocell Xi, pocell Xj, int i, int j) {
    pocell dF = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    pocell r = Xi - Xj;
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    float F = 2*(R_MIN - dist)*(R_MAX - dist) + powf(R_MAX - dist, 2);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    assert(dF.x == dF.x);  // For NaN f != f.

    dF += polarity_force(Xi, Xj)*0.2;
    return dF;
}

__device__ __managed__ nhoodint<pocell> potential = epithelium;


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(0.733333, X);
    for (int i = 0; i < N_CELLS; i++) {
        float dist = sqrtf(X[i].x*X[i].x + X[i].y*X[i].y + X[i].z*X[i].z);
        X[i].phi = atan2(X[i].y, X[i].x) + rand()/(RAND_MAX + 1.)*0.5;
        X[i].theta = acosf(X[i].z/dist) + rand()/(RAND_MAX + 1.)*0.5;
    }

    // Integrate cell positions
    VtkOutput output("epithelium");
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(X);
        output.write_polarity(X);
        if (time_step == N_TIME_STEPS) return 0;

        X.step(DELTA_T, potential);
    }
}
