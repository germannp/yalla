#include "../lib/dtypes.cuh"
#include "../lib/solvers.cuh"
#include "../lib/protrusions.cuh"
#include "minunit.cuh"


Solution<float3, 4, N2nSolver> bolls;
__device__ __managed__ Protrusions<4> prots;


__device__ float3 no_interaction(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    return dF;
}

__device__ auto d_no_interaction = &no_interaction;
auto h_no_interaction = get_device_object(d_no_interaction, 0);

void prots_forces(const float3* __restrict__ bolls, float3* dX) {
    intercalate<<<(4 + 32 - 1)/32, 32>>>(bolls, dX, prots);
}


const char* square_of_four() {
    bolls.h_X[0].x = 1;  bolls.h_X[0].y = 1;  bolls.h_X[0].z = 0;
    bolls.h_X[1].x = 1;  bolls.h_X[1].y = -1; bolls.h_X[1].z = 0;
    bolls.h_X[2].x = -1; bolls.h_X[2].y = -1; bolls.h_X[2].z = 0;
    bolls.h_X[3].x = -1; bolls.h_X[3].y = 1;  bolls.h_X[3].z = 0;
    bolls.memcpyHostToDevice();
    init_protrusions(prots);
    prots.links[0][0] = 0; prots.links[0][1] = 1;
    prots.links[1][0] = 1; prots.links[1][1] = 2;
    prots.links[2][0] = 2; prots.links[2][1] = 3;
    prots.links[3][0] = 3; prots.links[3][1] = 0;

    auto com_i = center_of_mass(bolls);
    for (auto i = 0; i < 500; i++) {
        bolls.step(0.1, h_no_interaction, prots_forces);
    }

    bolls.memcpyDeviceToHost();
    auto com_f = center_of_mass(bolls);
    MU_ASSERT("Momentum in square", MU_ISCLOSE(com_i.x, com_f.x));
    MU_ASSERT("Momentum in square", MU_ISCLOSE(com_i.y, com_f.y));
    MU_ASSERT("Momentum in square", MU_ISCLOSE(com_i.z, com_f.z));

    MU_ASSERT("Not close to origin in x", MU_ISCLOSE(bolls.h_X[0].x, 0));
    MU_ASSERT("Not close to origin in y", MU_ISCLOSE(bolls.h_X[0].y, 0));
    MU_ASSERT("Not close to origin in z", MU_ISCLOSE(bolls.h_X[0].z, 0));

    return NULL;
}


const char* all_tests() {
    MU_RUN_TEST(square_of_four);
    return NULL;
}

MU_RUN_SUITE(all_tests);
