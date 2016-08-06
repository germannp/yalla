#include "../lib/dtypes.cuh"
#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "minunit.cuh"


const int N_MAX = 1000;
const float L_0 = 1;

Solution<float3, N_MAX, N2nSolver> n2n;
Solution<float3, N_MAX, LatticeSolver> latt;


__device__ float3 spring(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    dF = r*(L_0 - dist)/dist;
    return dF;
}

__device__ auto d_spring = &spring;
auto h_spring = get_device_object(d_spring);

const char* test_n2n_tetrahedron() {
    n2n.set_n(4);
    uniform_sphere(L_0, n2n);
    auto com_i = center_of_mass(n2n);
    for (auto i = 0; i < 500; i++) {
        n2n.step(0.1, h_spring);
    }

    n2n.memcpyDeviceToHost();
    for (auto i = 1; i < 4; i++) {
        auto r = n2n.h_X[0] - n2n.h_X[i];
        auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        MU_ASSERT("Spring not relaxed in n2n tetrahedron", MU_ISCLOSE(dist, L_0));
    }

    auto com_f = center_of_mass(n2n);
    MU_ASSERT("Momentum in n2n tetrahedron", MU_ISCLOSE(com_i.x, com_f.x));
    MU_ASSERT("Momentum in n2n tetrahedron", MU_ISCLOSE(com_i.y, com_f.y));
    MU_ASSERT("Momentum in n2n tetrahedron", MU_ISCLOSE(com_i.z, com_f.z));

    return NULL;
}

const char* test_latt_tetrahedron() {
    latt.set_n(4);
    uniform_sphere(L_0, latt);
    auto com_i = center_of_mass(latt);
    for (auto i = 0; i < 500; i++) {
        latt.step(0.1, h_spring);
    }

    latt.memcpyDeviceToHost();
    for (auto i = 1; i < 4; i++) {
        auto r = float3{latt.h_X[0].x - latt.h_X[i].x, latt.h_X[0].y - latt.h_X[i].y,
            latt.h_X[0].z - latt.h_X[i].z};
        auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
        MU_ASSERT("Spring not relaxed in lattice tetrahedron", MU_ISCLOSE(dist, L_0));
    }

    auto com_f = center_of_mass(latt);
    MU_ASSERT("Momentum in lattice tetrahedron", MU_ISCLOSE(com_i.x, com_f.x));
    MU_ASSERT("Momentum in lattice tetrahedron", MU_ISCLOSE(com_i.y, com_f.y));
    MU_ASSERT("Momentum in lattice tetrahedron", MU_ISCLOSE(com_i.z, com_f.z));

    return NULL;
}


__device__ float3 cubic(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    if (i == j) return dF;

    auto r = Xi - Xj;
    auto dist = fminf(sqrtf(r.x*r.x + r.y*r.y + r.z*r.z), 1);
    auto F = 2*(0.6 - dist)*(1 - dist) + (1 - dist)*(1 - dist);
    dF = r*F/dist;
    return dF;
}

__device__ auto d_cubic = &cubic;
auto h_cubic = get_device_object(d_cubic, 0);

const char* test_compare_methods() {
    n2n.set_n(N_MAX);
    latt.set_n(N_MAX);
    uniform_sphere(0.733333, n2n);
    for (auto i = 0; i < N_MAX; i++) {
        latt.h_X[i].x = n2n.h_X[i].x;
        latt.h_X[i].y = n2n.h_X[i].y;
        latt.h_X[i].z = n2n.h_X[i].z;
    }
    latt.memcpyHostToDevice();
    n2n.step(0.5, h_cubic);
    latt.step(0.5, h_cubic);

    n2n.memcpyDeviceToHost();
    latt.memcpyDeviceToHost();
    for (auto i = 0; i < N_MAX; i++) {
        MU_ASSERT("Methods disagree", MU_ISCLOSE(n2n.h_X[i].x, latt.h_X[i].x));
        MU_ASSERT("Methods disagree", MU_ISCLOSE(n2n.h_X[i].y, latt.h_X[i].y));
        MU_ASSERT("Methods disagree", MU_ISCLOSE(n2n.h_X[i].z, latt.h_X[i].z));
    }

    return NULL;
}


__device__ float3 no_interaction(float3 Xi, float3 Xj, int i, int j) {
    float3 dF {0};
    return dF;
}

__device__ auto d_no_interaction = &no_interaction;
auto h_no_interaction = get_device_object(d_no_interaction, 0);

__global__ void push(float3* d_dX) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 1) return;

    d_dX[0] = float3{1, 0, 0};
}

void push_genforce(const float3* __restrict__ d_X, float3* d_dX) {
    push<<<1, 1>>>(d_dX);
}

const char* test_generic_forces() {
    n2n.h_X[0] = float3{0, 0, 0};
    n2n.memcpyHostToDevice();
    n2n.step(1, h_no_interaction, push_genforce);

    n2n.memcpyDeviceToHost();
    MU_ASSERT("N2n Generic force failed", MU_ISCLOSE(n2n.h_X[0].x, 1));
    MU_ASSERT("N2n Generic force failed", MU_ISCLOSE(n2n.h_X[0].y, 0));
    MU_ASSERT("N2n Generic force failed", MU_ISCLOSE(n2n.h_X[0].z, 0));

    latt.h_X[0] = float3{0, 0, 0};
    latt.memcpyHostToDevice();
    latt.step(1, h_no_interaction, push_genforce);

    latt.memcpyDeviceToHost();
    MU_ASSERT("Lattice Generic force failed", MU_ISCLOSE(latt.h_X[0].x, 1));
    MU_ASSERT("Lattice Generic force failed", MU_ISCLOSE(latt.h_X[0].y, 0));
    MU_ASSERT("Lattice Generic force failed", MU_ISCLOSE(latt.h_X[0].z, 0));

    return NULL;
}


template<int N_MAX>
__global__ void single_lattice(const Lattice<N_MAX>* __restrict__ d_lattice) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 1000) return;

    auto expected_cube = (LATTICE_SIZE*LATTICE_SIZE*LATTICE_SIZE)/2
        + (LATTICE_SIZE*LATTICE_SIZE)/2 + LATTICE_SIZE/2
        + i%10 + (i%100/10)*LATTICE_SIZE + (i/100)*LATTICE_SIZE*LATTICE_SIZE;
    D_ASSERT(d_lattice->d_cube_id[i] == expected_cube);
}

template<int N_MAX>
__global__ void double_lattice(const Lattice<N_MAX>* __restrict__ d_lattice) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= 1000 - 8) return;

    D_ASSERT(d_lattice->d_cube_id[i] == d_lattice->d_cube_id[i - i%8]);
}

const char* test_lattice_spacing() {
    for (auto i = 0; i < 10; i++) {
        for (auto j = 0; j < 10; j++) {
            for (auto k = 0; k < 10; k++) {
                latt.h_X[100*i + 10*j + k].x = k + 0.5;
                latt.h_X[100*i + 10*j + k].y = j + 0.5;
                latt.h_X[100*i + 10*j + k].z = i + 0.5;
            }
        }
    }
    latt.memcpyHostToDevice();

    latt.build_lattice(1);
    single_lattice<<<256, 4>>>(latt.d_lattice);

    latt.build_lattice(2);
    double_lattice<<<256, 4>>>(latt.d_lattice);
    cudaDeviceSynchronize();  // Wait for device to exit

    return NULL;
}


const char* all_tests() {
    MU_RUN_TEST(test_latt_tetrahedron);  // FIXME: Some orders brake tests!
    MU_RUN_TEST(test_n2n_tetrahedron);
    MU_RUN_TEST(test_compare_methods);
    MU_RUN_TEST(test_generic_forces);
    MU_RUN_TEST(test_lattice_spacing);
    return NULL;
}

MU_RUN_SUITE(all_tests);
