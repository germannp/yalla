// Links between bolls, to simulate protrusions, ...
#pragma once

#include <assert.h>
#include <curand_kernel.h>


__global__ void setup_rand_states(curandState* d_state, int n_states) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_states) curand_init(1337, i, 0, &d_state[i]);
}


struct Link { int a, b; };

template<int n_links>
class Links {
public:
    Link *h_link = (Link*)malloc(n_links*sizeof(Link));
    Link *d_link;
    int *h_n = (int*)malloc(sizeof(int));
    int *d_n;
    curandState *d_state;
    float strength;
    Links (float s = 1.f/5, int n_0 = n_links) {
        cudaMalloc(&d_link, n_links*sizeof(Link));
        cudaMalloc(&d_n, sizeof(int));
        cudaMalloc(&d_state, n_links*sizeof(curandState));
        for (auto i = 0; i < n_links; i++) {
            h_link[i].a = 0;
            h_link[i].b = 0;
        }
        *h_n = n_0;
        copy_to_device();
        setup_rand_states<<<(n_links + 32 - 1)/32, 32>>>(d_state, n_links);
        strength = s;
    }
    void set_d_n(int n) {
        assert(n <= n_links);
        cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    }
    int get_d_n() {
        int n;
        cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(n <= n_links);
        return n;
    }
    void copy_to_device() {
        assert(*h_n <= n_links);
        cudaMemcpy(d_link, h_link, n_links*sizeof(Link), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n, h_n, sizeof(int), cudaMemcpyHostToDevice);
    }
    void copy_to_host() {
        cudaMemcpy(h_link, d_link, n_links*sizeof(Link), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        assert(*h_n <= n_links);
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
template<int n_links, typename Pt = float3>
void linear_force(Links<n_links>& links, const Pt* __restrict__ d_X, Pt* d_dX) {
    link_force<<<(links.get_d_n() + 32 - 1)/32, 32>>>(d_X, d_dX, links.d_link,
        links.get_d_n(), links.strength);
}
