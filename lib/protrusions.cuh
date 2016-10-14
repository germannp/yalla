// Protrusions as constant forces between linked Pts
#pragma once

#include <assert.h>
#include <curand_kernel.h>


__global__ void setup_rand_states(curandState* d_state, int n_states) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_states) curand_init(1337, i, 0, &d_state[i]);
}


struct Link { int a, b; };

template<int N_LINKS>
class Protrusions {
public:
    Link *h_link = (Link*)malloc(N_LINKS*sizeof(Link));
    Link *d_link;
    int *h_n = (int*)malloc(sizeof(int));
    int *d_n;
    curandState *d_state;
    float strength;
    Protrusions (float s = 1.f/5, int n = N_LINKS) {
        cudaMalloc(&d_link, N_LINKS*sizeof(Link));
        cudaMalloc(&d_n, sizeof(int));
        cudaMalloc(&d_state, N_LINKS*sizeof(curandState));
        for (auto i = 0; i < N_LINKS; i++) {
            h_link[i].a = 0;
            h_link[i].b = 0;
        }
        *h_n = n;
        memcpyHostToDevice();
        setup_rand_states<<<(N_LINKS + 32 - 1)/32, 32>>>(d_state, N_LINKS);
        strength = s;
    }
    void set_d_n(int n) {
        assert(n <= N_LINKS);
        cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    }
    int get_d_n() {
        int n;
        cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(n <= N_LINKS);
        return n;
    }
    void memcpyHostToDevice() {
        assert(*h_n <= N_LINKS);
        cudaMemcpy(d_link, h_link, N_LINKS*sizeof(Link), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n, h_n, sizeof(int), cudaMemcpyHostToDevice);
    }
    void memcpyDeviceToHost() {
        cudaMemcpy(h_link, d_link, N_LINKS*sizeof(Link), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(*h_n <= N_LINKS);
    }
};


template<typename Pt>
__global__ void link_force(const Pt* __restrict__ d_X, Pt* d_dX,
        const Link* __restrict__ d_link, int n_links, float strength) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_links) return;

    auto j = d_link[i].a;
    auto k = d_link[i].b;
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

// Passing pointers to non-static members needs some std::bind (or std::mem_func),
// see http://stackoverflow.com/questions/37924781/. I prefer binding a seperate function.
template<int N_LINKS, typename Pt = float3>
void link_forces(Protrusions<N_LINKS>& links, const Pt* __restrict__ d_X, Pt* d_dX) {
    link_force<<<(links.get_d_n() + 32 - 1)/32, 32>>>(d_X, d_dX, links.d_link,
        links.get_d_n(), links.strength);
}
