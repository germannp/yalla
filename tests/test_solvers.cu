#include "../include/dtypes.cuh"
#include "../include/solvers.cuh"
#include "../include/inits.cuh"
#include "minunit.cuh"


__device__ float4 oscillator(float4 Xi, float4 r, float dist, int i, int j) {
    float4 dF {0};
    if (i == j) return dF;

    if (i == 0) return Xi - r;

    return - (Xi - r);
}

const char* test_oscillation() {
    Solution<float4, 2, N2n_solver> oscillation;
    oscillation.h_X[0].w = 1;
    oscillation.h_X[1].w = 0;
    oscillation.copy_to_device();

    auto n_steps = 100;
    for (auto i = 0; i < n_steps; i++) {
        oscillation.take_step<oscillator>(2*M_PI/n_steps);
        oscillation.copy_to_host();
        MU_ASSERT("Oscillator off circle", MU_ISCLOSE(
            powf(oscillation.h_X[0].w, 2) + powf(oscillation.h_X[1].w, 2), 1));
    }
    oscillation.copy_to_host();
    MU_ASSERT("Oscillator final cosine", MU_ISCLOSE(oscillation.h_X[0].w, 1));
    // The sine is substantially less precise ;-)

    return NULL;
}


const auto L_0 = 0.5;

__device__ float3 clipped_spring(float3 Xi, float3 r, float dist, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    if (dist >= 1) return dF;

    dF = r*(L_0 - dist)/dist;
    return dF;
}

const char* test_n2n_tetrahedron() {
    Solution<float3, 4, N2n_solver> n2n;
    uniform_sphere(L_0, n2n);
    for (auto i = 0; i < 500; i++) {
        n2n.take_step<clipped_spring>(0.1);
    }

    n2n.copy_to_host();
    for (auto i = 1; i < 4; i++) {
        auto r = n2n.h_X[0] - n2n.h_X[i];
        auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        MU_ASSERT("Spring not relaxed in n2n tetrahedron", MU_ISCLOSE(dist, L_0));
    }

    return NULL;
}

const char* test_latt_tetrahedron() {
    Solution<float3, 4, Lattice_solver> latt;
    uniform_sphere(L_0, latt);
    for (auto i = 0; i < 500; i++) {
        latt.take_step<clipped_spring>(0.1);
    }

    latt.copy_to_host();
    for (auto i = 1; i < 4; i++) {
        auto r = float3{latt.h_X[0].x - latt.h_X[i].x, latt.h_X[0].y - latt.h_X[i].y,
            latt.h_X[0].z - latt.h_X[i].z};
        auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        MU_ASSERT("Spring not relaxed in lattice tetrahedron", MU_ISCLOSE(dist, L_0));
    }

    return NULL;
}

const auto n_max = 50;

const char* test_compare_methods() {
    Solution<float3, n_max, N2n_solver> n2n;
    Solution<float3, n_max, Lattice_solver> latt;
    uniform_sphere(0.733333, n2n);
    for (auto i = 0; i < n_max; i++) {
        latt.h_X[i].x = n2n.h_X[i].x;
        latt.h_X[i].y = n2n.h_X[i].y;
        latt.h_X[i].z = n2n.h_X[i].z;
    }
    latt.copy_to_device();
    n2n.take_step<clipped_spring>(0.5);
    latt.take_step<clipped_spring>(0.5);

    n2n.copy_to_host();
    latt.copy_to_host();
    for (auto i = 0; i < n_max; i++) {
        MU_ASSERT("Methods disagree", MU_ISCLOSE(n2n.h_X[i].x, latt.h_X[i].x));
        MU_ASSERT("Methods disagree", MU_ISCLOSE(n2n.h_X[i].y, latt.h_X[i].y));
        MU_ASSERT("Methods disagree", MU_ISCLOSE(n2n.h_X[i].z, latt.h_X[i].z));
    }

    return NULL;
}


__global__ void push_cell(float3* d_dX) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i != 0) return;

    d_dX[1] = float3{1, 0, 0};
}

void push(const float3* __restrict__ d_X, float3* d_dX) {
    push_cell<<<1, 1>>>(d_dX);
}

const char* test_generic_forces() {
    Solution<float3, 2, N2n_solver> n2n;
    n2n.h_X[0] = float3{0, 0, 10};
    n2n.h_X[1] = float3{0, 0, 0};
    n2n.copy_to_device();
    n2n.take_step<clipped_spring>(1, push);

    n2n.copy_to_host();
    MU_ASSERT("N2n Generic force failed in x", MU_ISCLOSE(n2n.h_X[1].x, 1));
    MU_ASSERT("N2n Generic force failed in y", MU_ISCLOSE(n2n.h_X[1].y, 0));
    MU_ASSERT("N2n Generic force failed in z", MU_ISCLOSE(n2n.h_X[1].z, 0));

    Solution<float3, 2, Lattice_solver> latt;
    latt.h_X[0] = float3{0, 0, 10};
    latt.h_X[1] = float3{0, 0, 0};
    latt.copy_to_device();
    latt.take_step<clipped_spring>(1, push);

    latt.copy_to_host();
    MU_ASSERT("Lattice Generic force failed in x", MU_ISCLOSE(latt.h_X[1].x, 1));
    MU_ASSERT("Lattice Generic force failed in y", MU_ISCLOSE(latt.h_X[1].y, 0));
    MU_ASSERT("Lattice Generic force failed in z", MU_ISCLOSE(latt.h_X[1].z, 0));

    return NULL;
}


template<int n_max>
__global__ void single_lattice(const Lattice<n_max>* __restrict__ d_lattice) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 1000) return;

    auto expected_cube = (LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE)/2
        + (LATTICE_SIZE*LATTICE_SIZE)/2 + LATTICE_SIZE/2
        + i%10 + (i%100/10)*LATTICE_SIZE + (i/100)*LATTICE_SIZE*LATTICE_SIZE;
    D_ASSERT(d_lattice->d_cube_id[i] == expected_cube);
}

template<int n_max>
__global__ void double_lattice(const Lattice<n_max>* __restrict__ d_lattice) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 1000 - 8) return;

    D_ASSERT(d_lattice->d_cube_id[i] == d_lattice->d_cube_id[i - i%8]);
}

const char* test_lattice_spacing() {
    Solution<float3, 1000, Lattice_solver> latt;
    for (auto i = 0; i < 10; i++) {
        for (auto j = 0; j < 10; j++) {
            for (auto k = 0; k < 10; k++) {
                latt.h_X[100*i + 10*j + k].x = k + 0.5;
                latt.h_X[100*i + 10*j + k].y = j + 0.5;
                latt.h_X[100*i + 10*j + k].z = i + 0.5;
            }
        }
    }
    latt.copy_to_device();

    latt.build_lattice(1);
    single_lattice<<<256, 4>>>(latt.d_lattice);

    latt.build_lattice(2);
    double_lattice<<<256, 4>>>(latt.d_lattice);
    cudaDeviceSynchronize();  // Wait for device to exit

    return NULL;
}


const char* all_tests() {
    MU_RUN_TEST(test_oscillation);
    MU_RUN_TEST(test_n2n_tetrahedron);
    MU_RUN_TEST(test_latt_tetrahedron);
    MU_RUN_TEST(test_compare_methods);
    MU_RUN_TEST(test_generic_forces);
    MU_RUN_TEST(test_lattice_spacing);
    return NULL;
}

MU_RUN_SUITE(all_tests);
