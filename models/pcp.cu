// Simulate planer cell polarity aligned by gradient
#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/vtk.cuh"


const auto r_max = 1;
const auto r_min = 0.6;
const auto D = 0.5f;
const auto n_cells = 50;
const auto n_time_steps = 300;
const auto dt = 0.1;

MAKE_PT(Po_cell4, x, y, z, w, phi, theta);


// Cubic potential plus Heisenberg biased by w
__device__ Po_cell4 pairwise_interaction(Po_cell4 Xi, Po_cell4 Xj, int i, int j) {
    Po_cell4 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = norm3df(r.x, r.y, r.z);
    if (dist > r_max) return dF;

    auto F = 2*(r_min - dist)*(r_max - dist) + powf(r_max - dist, 2);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;
    dF.w = i == 0 ? 0 : -r.w*D;

    // Heisenberg potential U = - (n_i . n_j)
    auto sin_Xi_theta = sinf(Xi.theta);
    if (fabs(sin_Xi_theta) > 1e-10)
        dF.phi = - sinf(Xj.theta)*sinf(Xi.phi - Xj.phi)/sin_Xi_theta;
    dF.theta = cosf(Xi.theta)*sinf(Xj.theta)*cosf(Xi.phi - Xj.phi) -
        sinf(Xi.theta)*cosf(Xj.theta);

    // U = - Xj.w*(n_i . r_ij/r) to bias along w
    auto r_phi = atan2(-r.y, -r.x);
    auto r_theta = acosf(-r.z/dist);
    if (fabs(sin_Xi_theta) > 1e-10)
        dF.phi += - Xj.w*sinf(r_theta)*sinf(Xi.phi - r_phi)/sin_Xi_theta;
    dF.theta += Xj.w*(cosf(Xi.theta)*sinf(r_theta)*cosf(Xi.phi - r_phi) -
        sinf(Xi.theta)*cosf(r_theta));

    return dF;
}

#include "../lib/solvers.cuh"


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<Po_cell4, n_cells, Lattice_solver> bolls;
    uniform_sphere(0.5, bolls);
    for (auto i = 0; i < n_cells; i++) {
        bolls.h_X[i].w = (i == 0);
        bolls.h_X[i].phi = 2.*M_PI*rand()/(RAND_MAX + 1.);
        bolls.h_X[i].theta = acos(2.*rand()/(RAND_MAX + 1.) - 1.);
    }
    bolls.copy_to_device();

    // Integrate cell positions
    Vtk_output output("pcp");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step(dt);
        output.write_positions(bolls);
        output.write_field(bolls);
        output.write_polarity(bolls);
    }

    return 0;
}
