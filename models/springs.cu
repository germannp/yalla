// Integrate N-body problem with springs between all bodies
#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/vtk.cuh"


const auto L_0 = 0.5f;  // Relaxed spring length
const auto n_cells = 800u;
const auto n_time_steps = 100u;
const auto dt = 0.001f;


__device__ float3 pairwise_interaction(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = norm3df(r.x, r.y, r.z);
    dF = r*(L_0 - dist)/dist;  // Spring force
    return dF;
}

#include "../lib/solvers.cuh"  // pairwise_interaction must be defined before


int main(int argc, const char* argv[]) {
    // Prepare initial state
    Solution<float3, n_cells, N2n_solver> bolls;
    uniform_sphere(L_0, bolls);

    // Integrate positions
    Vtk_output output("springs");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step(dt);            // Ordering to start writing during calculation,
        output.write_positions(bolls);  // use thread for full concurency.
    }

    return 0;
}
