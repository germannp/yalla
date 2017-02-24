#include <functional>

#include "../lib/dtypes.cuh"
#include "../lib/solvers.cuh"
#include "../lib/links.cuh"
#include "minunit.cuh"


__device__ float3 no_pw_int(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    return dF;
}


const char* square_of_four() {
    Solution<float3, 4, N2n_solver> bolls;
    Links<4> links;
    auto forces = std::bind(linear_force<4>, links,
        std::placeholders::_1, std::placeholders::_2);

    bolls.h_X[0].x = 1;  bolls.h_X[0].y = 1;  bolls.h_X[0].z = 0;
    bolls.h_X[1].x = 1;  bolls.h_X[1].y = -1; bolls.h_X[1].z = 0;
    bolls.h_X[2].x = -1; bolls.h_X[2].y = -1; bolls.h_X[2].z = 0;
    bolls.h_X[3].x = -1; bolls.h_X[3].y = 1;  bolls.h_X[3].z = 0;
    bolls.copy_to_device();
    links.h_link[0].a = 0; links.h_link[0].b = 1;
    links.h_link[1].a = 1; links.h_link[1].b = 2;
    links.h_link[2].a = 2; links.h_link[2].b = 3;
    links.h_link[3].a = 3; links.h_link[3].b = 0;
    links.copy_to_device();

    auto com_i = center_of_mass(bolls);
    for (auto i = 0; i < 500; i++) {
        bolls.take_step<no_pw_int>(0.1, forces);
    }

    bolls.copy_to_host();
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
