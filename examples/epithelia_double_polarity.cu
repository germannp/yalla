#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1.0f;
const auto n_cells = 300;
const auto n_time_steps = 500;
const auto skip_step = 5 ;
const auto dt = 0.1;

// Epithelial cells with two polarity vectors,
// spherical coordinates for one are (theta, phi)
// and one for the othre are (iota, chi).
MAKE_PT(Pcp_epi, theta, phi, iota, chi);

__device__ Pcp_epi force_A(
    Pcp_epi Xi, Pcp_epi r, float dist, int i, int j)
{
    Pcp_epi dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto F = fmaxf(0.8 - dist, 0)*1.0 - fmaxf(dist - 0.8, 0)*1.5f;

    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF += bending_force(Xi, r, dist)*0.3;
    return dF;
}


__device__ Pcp_epi force_B(
    Pcp_epi Xi, Pcp_epi r, float dist, int i, int j)
{
    Pcp_epi dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto F = fmaxf(0.8 - dist, 0)*1.0 - fmaxf(dist - 0.8, 0)*1.5f;

    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF += bending_force<Pcp_epi, &Pcp_epi::iota, &Pcp_epi::chi>(Xi, r, dist)*0.3;

    return dF;
}

int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Pcp_epi, Grid_solver> cells{n_cells};

    random_sphere(0.8, cells);
    for (auto i = 0; i < n_cells; i++) {
        auto dist = sqrtf(cells.h_X[i].x * cells.h_X[i].x +
                          cells.h_X[i].y * cells.h_X[i].y +
                          cells.h_X[i].z * cells.h_X[i].z);
        cells.h_X[i].theta = acosf(cells.h_X[i].z/dist);
        cells.h_X[i].phi = atan2(cells.h_X[i].y, cells.h_X[i].x);

        cells.h_X[i].iota = acosf(0.0);
        cells.h_X[i].chi = atan2(0.0, 1.0);

    }
    cells.copy_to_device();

    // Integrate cell positions
    Vtk_output output{"epithelia_double_polarity", "output/", true};

    for (auto time_step = 0; time_step <= n_time_steps/2; time_step++) {

        cells.take_step<force_A>(dt);

        if(time_step%skip_step == 0){
            cudaDeviceSynchronize();
            cells.copy_to_host();
            output.write_positions(cells);
            output.write_polarity(cells);
            output.write_polarity<Pcp_epi, &Pcp_epi::iota, &Pcp_epi::chi>(cells, "pcp");
        }
    }

    for (auto time_step = 1; time_step <= n_time_steps/2; time_step++) {

        cells.take_step<force_B>(dt);

        if(time_step%skip_step == 0){
            cudaDeviceSynchronize();
            cells.copy_to_host();
            output.write_positions(cells);
            output.write_polarity(cells);
            output.write_polarity<Pcp_epi, &Pcp_epi::iota, &Pcp_epi::chi>(cells, "pcp");

        }
    }
    return 0;
}
