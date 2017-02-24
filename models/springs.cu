// Integrate N-body problem with springs between all bodies
#include "../lib/dtypes.cuh"
#include "../lib/solvers.cuh"
#include "../lib/inits.cuh"
#include "../lib/vtk.cuh"


const auto L_0 = 0.5f;  // Relaxed spring length
const auto n_cells = 800u;
const auto n_time_steps = 100u;
const auto dt = 0.001f;


__device__ float3 spring_force(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = norm3df(r.x, r.y, r.z);
    dF = r*(L_0 - dist)/dist;
    return dF;
}


int main(int argc, const char* argv[]) {
    // Prepare initial state
    Solution<float3, n_cells, N2n_solver> bolls;
    uniform_sphere(L_0, bolls);

    // Integrate positions
    Vtk_output output("springs");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.take_step<spring_force>(dt);  // Ordering to write during calculation,
        output.write_positions(bolls);      // use thread for full concurency.
    }

    return 0;
}
