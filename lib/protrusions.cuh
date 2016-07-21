// Protrusions as constant forces between linked Pts
#include <cassert>
#include <curand_kernel.h>


__global__ void setup_rand_states(curandState* d_state, int n_states) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_states) curand_init(1337, i, 0, &d_state[i]);
}


struct Link { int a, b; };

template<int N_LINKS>
struct Protrusions {
    Link *h_cell_id = (Link*)malloc(N_LINKS*sizeof(Link));
    Link *d_cell_id;
    curandState *d_state;
    Protrusions () {
        cudaMalloc(&d_cell_id, N_LINKS*sizeof(Link));
        cudaMalloc(&d_state, N_LINKS*sizeof(curandState));
        for (auto i = 0; i < N_LINKS; i++) {
            h_cell_id[i].a = 0;
            h_cell_id[i].b = 0;
        }
        memcpyHostToDevice();
        setup_rand_states<<<(N_LINKS + 32 - 1)/32, 32>>>(d_state, N_LINKS);
    }
    void memcpyHostToDevice() {
        cudaMemcpy(d_cell_id, h_cell_id, N_LINKS*sizeof(Link), cudaMemcpyHostToDevice);
    }
    void memcpyDeviceToHost() {
        cudaMemcpy(h_cell_id, d_cell_id, N_LINKS*sizeof(Link), cudaMemcpyDeviceToHost);
    }
};


template<typename Pt>
__global__ void link_force(const Pt* __restrict__ d_X, Pt* d_dX,
        const Link* __restrict__ d_cell_id, int n_links, float strength = 1.f/5) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_links) return;

    auto j = d_cell_id[i].a;
    auto k = d_cell_id[i].b;
    if (j == k) return;

    auto r = d_X[j] - d_X[k];
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    atomicAdd(&d_dX[j].x, -strength*r.x/dist);
    atomicAdd(&d_dX[j].y, -strength*r.y/dist);
    atomicAdd(&d_dX[j].z, -strength*r.z/dist);
    atomicAdd(&d_dX[k].x, strength*r.x/dist);
    atomicAdd(&d_dX[k].y, strength*r.y/dist);
    atomicAdd(&d_dX[k].z, strength*r.z/dist);
}
