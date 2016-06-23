#include "../lib/dtypes.cuh"
#include "../lib/solvers.cuh"
#include "../lib/protrusions.cuh"
#include "minunit.cuh"


__device__ __managed__ Solution<float3, 4, N2nSolver> X;
__device__ __managed__ Protrusions<4> prots;


__device__ float3 no_interaction(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    return dF;
}

__device__ __managed__ auto d_potential = no_interaction;


void prots_forces(const float3* __restrict__ X, float3* dX) {
    intercalate<<<(4 + 32 - 1)/32, 32>>>(X, dX, prots);
    cudaDeviceSynchronize();
}


const char* square_of_four() {
    X[0].x = 1;  X[0].y = 1;  X[0].z = 0;
    X[1].x = 1;  X[1].y = -1; X[1].z = 0;
    X[2].x = -1; X[2].y = -1; X[2].z = 0;
    X[3].x = -1; X[3].y = 1;  X[3].z = 0;
    init_protrusions(prots);
    prots.links[0][0] = 0; prots.links[0][1] = 1;
    prots.links[1][0] = 1; prots.links[1][1] = 2;
    prots.links[2][0] = 2; prots.links[2][1] = 3;
    prots.links[3][0] = 3; prots.links[3][1] = 0;

    auto com_i = center_of_mass(X);
    for (auto i = 0; i < 500; i++) {
        X.step(0.1, d_potential, prots_forces);
    }

    auto com_f = center_of_mass(X);
    MU_ASSERT("Momentum in square", MU_ISCLOSE(com_i.x, com_f.x));
    MU_ASSERT("Momentum in square", MU_ISCLOSE(com_i.y, com_f.y));
    MU_ASSERT("Momentum in square", MU_ISCLOSE(com_i.z, com_f.z));

    MU_ASSERT("Not close to origin in x", MU_ISCLOSE(X[0].x, 0));
    MU_ASSERT("Not close to origin in y", MU_ISCLOSE(X[0].y, 0));
    MU_ASSERT("Not close to origin in z", MU_ISCLOSE(X[0].z, 0));

    return NULL;
}


const char* all_tests() {
    MU_RUN_TEST(square_of_four);
    return NULL;
}

MU_RUN_SUITE(all_tests);
