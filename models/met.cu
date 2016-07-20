// Simulate a mesenchyme-to-epithelium transition
#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"
#include "../lib/epithelium.cuh"


const auto R_MAX = 1;
const auto R_MIN = 0.6;
const auto N_CELLS = 250;
const auto N_TIME_STEPS = 100;
const auto DELTA_T = 0.1;

Solution<pocell, N_CELLS, LatticeSolver> bolls;


// Cubic potential plus k*(n_i . r_ij/r)^2/2 for all r_ij <= R_MAX
__device__ pocell epithelium(pocell Xi, pocell Xj, int i, int j) {
    pocell dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    auto F = 2*(R_MIN - dist)*(R_MAX - dist) + powf(R_MAX - dist, 2);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    dF += polarity_force(Xi, Xj)*0.2;
    return dF;
}

__device__ auto d_epithelium = &epithelium;
auto h_epithelium = get_device_object(d_epithelium, 0);


int main(int argc, char const *argv[]) {
    // Prepare initial state
    uniform_sphere(0.733333, bolls);
    for (auto i = 0; i < N_CELLS; i++) {
        auto dist = sqrtf(bolls.h_X[i].x*bolls.h_X[i].x + bolls.h_X[i].y*bolls.h_X[i].y
            + bolls.h_X[i].z*bolls.h_X[i].z);
        bolls.h_X[i].phi = atan2(bolls.h_X[i].y, bolls.h_X[i].x) + rand()/(RAND_MAX + 1.)*0.5;
        bolls.h_X[i].theta = acosf(bolls.h_X[i].z/dist) + rand()/(RAND_MAX + 1.)*0.5;
    }
    bolls.memcpyHostToDevice();

    // Integrate cell positions
    VtkOutput output("epithelium");
    for (auto time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        bolls.memcpyDeviceToHost();
        bolls.step(DELTA_T, h_epithelium);
        output.write_positions(bolls);
        output.write_polarity(bolls);
    }

    return 0;
}
