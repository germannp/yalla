// Simulate planer cell polarity aligned by gradient
#include "../lib/dtypes.cuh"
#include "../lib/solvers.cuh"
#include "../lib/inits.cuh"
#include "../lib/vtk.cuh"


const auto r_max = 1;
const auto r_min = 0.6;
const auto D = 0.5f;
const auto n_cells = 500;
const auto n_time_steps = 300;
const auto dt = 0.025;

MAKE_PT(Po_cell4, x, y, z, w, phi, theta);


__device__ Po_cell4 biased_heisenberg(Po_cell4 Xi, Po_cell4 Xj, int i, int j) {
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

    // U_PCP = Σ(n_i . n_j)^2/2
    auto prod = sinf(Xi.theta)*sinf(Xj.theta)*cosf(Xi.phi - Xj.phi)
        + cosf(Xi.theta)*cosf(Xj.theta);
    auto sin_Xi_theta = sinf(Xi.theta);
    if (fabs(sin_Xi_theta) > 1e-10)
        dF.phi = - prod*sinf(Xj.theta)*sinf(Xi.phi - Xj.phi)/sin_Xi_theta;
    dF.theta = prod*(cosf(Xi.theta)*sinf(Xj.theta)*cosf(Xi.phi - Xj.phi) -
        sinf(Xi.theta)*cosf(Xj.theta));

    if (r.w > 0) return dF;

    // U_WNT = - ΣXj.w*(n_i . r_ij/r)^2/2 to bias along w
    auto r_phi = atan2(-r.y, -r.x);
    auto r_theta = acosf(-r.z/dist);
    prod = sinf(Xi.theta)*sinf(r_theta)*cosf(Xi.phi - r_phi)
        + cosf(Xi.theta)*cosf(r_theta);
    if (fabs(sin_Xi_theta) > 1e-10)
        dF.phi += - Xj.w*prod*sinf(r_theta)*sinf(Xi.phi - r_phi)/sin_Xi_theta;
    dF.theta += Xj.w*prod*(cosf(Xi.theta)*sinf(r_theta)*cosf(Xi.phi - r_phi) -
        sinf(Xi.theta)*cosf(r_theta));

    return dF;
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    Solution<Po_cell4, n_cells, Lattice_solver> bolls;
    for (auto i = 0; i < n_cells; i++) {
        bolls.h_X[i].w = (i == 0)*10;
        bolls.h_X[i].phi = 2.*M_PI*rand()/(RAND_MAX + 1.);
        bolls.h_X[i].theta = acos(2.*rand()/(RAND_MAX + 1.) - 1.);
    }
    uniform_sphere(0.5, bolls);

    // Integrate cell positions
    Vtk_output output("pcp");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step<biased_heisenberg>(dt);
        output.write_positions(bolls);
        output.write_field(bolls);
        output.write_polarity(bolls);
    }

    return 0;
}
