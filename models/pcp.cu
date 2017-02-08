// Simulate planer cell polarity
#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/vtk.cuh"
#include "../lib/polarity.cuh"


const auto r_max = 1;
const auto r_min = 0.6;
const auto n_cells = 250;
const auto n_time_steps = 100;
const auto dt = 0.1;


// Cubic potential plus k*(n_i . n_j)^2/2 for all r_ij <= r_max
__device__ Po_cell pairwise_interaction(Po_cell Xi, Po_cell Xj, int i, int j) {
    Po_cell dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > r_max) return dF;

    auto F = 2*(r_min - dist)*(r_max - dist) + powf(r_max - dist, 2);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    // n1 . n2 = sin(t1)*sin(t2)*cos(p1 - p2) + cos(t1)*cos(t2)
    dF.phi = sinf(Xi.theta)*sinf(Xj.theta)*sinf(Xi.phi - Xj.phi);
    dF.theta = - cosf(Xi.theta)*sinf(Xj.theta)*cosf(Xi.phi - Xj.phi) -
        sinf(Xi.theta)*cosf(Xj.theta);

    return dF;
}

#include "../lib/solvers.cuh"


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<Po_cell, n_cells, Lattice_solver> bolls;
    uniform_sphere(0.5, bolls);
    for (auto i = 0; i < n_cells; i++) {
        bolls.h_X[i].phi = 2*M_PI*rand()/(RAND_MAX + 1.);
        bolls.h_X[i].theta = M_PI*rand()/(RAND_MAX + 1.);
    }
    bolls.copy_to_device();

    // Integrate cell positions
    Vtk_output output("pcp");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step(dt);
        output.write_positions(bolls);
        output.write_polarity(bolls);
    }

    return 0;
}
