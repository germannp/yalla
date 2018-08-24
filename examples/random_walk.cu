// Simulate randomly migrating cell
#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/polarity.cuh"
#include "../include/solvers.cuh"
#include "../include/utils.cuh"
#include "../include/vtk.cuh"


const auto r_max = 1;
const auto n_cells = 500;
const auto n_time_steps = 150;
const auto dt = 0.05;


__device__ Po_cell relu_w_migration(
    Po_cell Xi, Po_cell r, float dist, int i, int j)
{
    Po_cell dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    auto F = fmaxf(0.7 - dist, 0) * 2 - fmaxf(dist - 0.8, 0);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    dF += migration_force(Xi, r, dist);
    return dF;
}


__global__ void update_polarities(Po_cell* d_X, curandState* d_state)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i != 0) return;

    // Pick random perturbation in cone around z axis
    Polarity perturbation{curand_normal(&d_state[i]),
        2.f * static_cast<float>(M_PI) * curand_uniform(&d_state[i])};

    // Rotate perturbation such that z axis would be in direction of migration
    auto dir = pol_to_float3(perturbation);
    auto u_phi = d_X[i].phi + M_PI / 2.;
    float3 u{cosf(u_phi), sinf(u_phi), 0};
    auto sin_theta = sinf(d_X[i].theta);
    auto cos_theta = cosf(d_X[i].theta);
    float3 new_dir;
    new_dir.x = (cos_theta + u.x * u.x * (1 - cos_theta)) * dir.x +
                u.x * u.y * (1 - cos_theta) * dir.y + u.y * sin_theta * dir.z;
    new_dir.y = u.x * u.y * (1 - cos_theta) * dir.x +
                (cos_theta + u.y * u.y * (1 - cos_theta)) * dir.y -
                u.x * sin_theta * dir.z;
    new_dir.z =
        -u.y * sin_theta * dir.x + u.x * sin_theta * dir.y + cos_theta * dir.z;
    auto new_polarity = pt_to_pol(new_dir);
    d_X[i].theta = new_polarity.theta;
    d_X[i].phi = new_polarity.phi;
}


int main(int argc, const char* argv[])
{
    // Prepare initial state
    Solution<Po_cell, Tile_solver> cells{n_cells};
    relaxed_sphere(0.75, cells);
    cells.h_X[0] = Po_cell{0};
    cells.h_X[0].phi = 0.01;
    cells.copy_to_device();
    curandState* d_state;
    cudaMalloc(&d_state, n_cells * sizeof(curandState));
    auto seed = time(NULL);
    setup_rand_states<<<(n_cells + 128 - 1) / 128, 128>>>(
        n_cells, seed, d_state);

    // Integrate cell positions
    Vtk_output output{"random_walk"};
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        cells.copy_to_host();
        update_polarities<<<(n_cells + 32 - 1) / 32, 32>>>(cells.d_X, d_state);
        cells.take_step<relu_w_migration>(dt);
        output.write_positions(cells);
        output.write_polarity(cells);
    }

    return 0;
}
