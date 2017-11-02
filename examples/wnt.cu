// Simulate polarities aligining with gradient
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto D = 1;
const auto n_cells = 61;
const auto n_time_steps = 150;
const auto dt = 0.025;

MAKE_PT(Po_cell4, w, theta, phi);


__device__ Po_cell4 diffusion(Po_cell4 Xi, Po_cell4 r, float dist, int i, int j)
{
    Po_cell4 dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    dF.w = i == 11 ? 0 : -r.w * D;
    if (r.w > 0) return dF;

    // U_WNT = - ΣXj.w*(n_i . r_ij/r)^2/2 to bias along w
    Polarity rhat{acosf(-r.z / dist), atan2(-r.y, -r.x)};
    dF += (Xi.w - r.w) * pcp_force(Xi, rhat);

    return dF;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Po_cell4, n_cells, Tile_solver> bolls;
    regular_hexagon(0.75, bolls);
    for (auto i = 0; i < n_cells; i++) {
        if (i == 11) {
            bolls.h_X[i].w = 50;
        } else {
            auto r = bolls.h_X[i] - bolls.h_X[11];  // Tilt polarities towards
            bolls.h_X[i].theta = 0.01;              // source to end w/ all
            bolls.h_X[i].phi = atan2(-r.y, -r.x);   // pointing the same way.
        }
    }
    bolls.copy_to_device();

    // Integrate cell positions
    Vtk_output output("wnt");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step<diffusion>(dt);
        output.write_positions(bolls);
        output.write_polarity(bolls);
        output.write_field(bolls);
    }

    return 0;
}
