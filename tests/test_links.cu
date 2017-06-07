#include <functional>

#include "../include/dtypes.cuh"
#include "../include/solvers.cuh"
#include "../include/links.cuh"
#include "minunit.cuh"


__device__ float3 no_pw_int(float3 Xi, float3 r, float dist, int i, int j) {
    float3 dF {0};
    return dF;
}


const char* square_of_four() {
    Solution<float3, 4, N2n_solver> bolls;
    Links<4> links;
    auto forces = std::bind(link_forces<4>, links,
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

    for (auto i = 0; i < 500; i++) {
        bolls.take_step<no_pw_int>(0.1, forces);
    }

    bolls.copy_to_host();

    MU_ASSERT("Not close to each other in x", MU_ISCLOSE(bolls.h_X[0].x, bolls.h_X[1].x));
    MU_ASSERT("Not close to each other in y", MU_ISCLOSE(bolls.h_X[1].y, bolls.h_X[2].y));
    MU_ASSERT("Not close to each other in z", MU_ISCLOSE(bolls.h_X[2].z, bolls.h_X[3].z));

    return NULL;
}


const char* all_tests() {
    MU_RUN_TEST(square_of_four);
    return NULL;
}

MU_RUN_SUITE(all_tests);
