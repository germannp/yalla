// Links between bolls, to simulate protrusions, ...
#pragma once

#include <assert.h>
#include <curand_kernel.h>
#include <functional>


__global__ void setup_rand_states(curandState* d_state, int n_states) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_states) curand_init(1337, i, 0, &d_state[i]);
}


struct Link { int a, b; };

using Check_link = std::function<bool (int a, int b)>;

bool every_link(int a, int b) { return true; }

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
        *h_n = n_0;
        set_d_n(n_0);
        reset();
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
    void reset(Check_link check = every_link) {
        copy_to_host();
        for (auto i = 0; i < n_links; i++) {
            if (!check(h_link[i].a, h_link[i].b)) continue;

            h_link[i].a = 0;
            h_link[i].b = 0;
        }
        copy_to_device();
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
using Link_force = void (const Pt* __restrict__ d_X, const int a, const int b,
    const float strength, Pt* d_dX);

template<typename Pt>
__device__ void linear_force(const Pt* __restrict__ d_X, const int a, const int b,
        const float strength, Pt* d_dX) {
    auto r = d_X[a] - d_X[b];
    auto dist = norm3df(r.x, r.y, r.z);

    atomicAdd(&d_dX[a].x, -strength*r.x/dist);
    atomicAdd(&d_dX[a].y, -strength*r.y/dist);
    atomicAdd(&d_dX[a].z, -strength*r.z/dist);
    atomicAdd(&d_dX[b].x, strength*r.x/dist);
    atomicAdd(&d_dX[b].y, strength*r.y/dist);
    atomicAdd(&d_dX[b].z, strength*r.z/dist);
}

template<typename Pt, Link_force<Pt> force>
__global__ void link(const Pt* __restrict__ d_X, Pt* d_dX,
        const Link* __restrict__ d_link, int n_links, float strength) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_links) return;

    auto a = d_link[i].a;
    auto b = d_link[i].b;
    if (a == b) return;

    force(d_X, a, b, strength, d_dX);
}

// Passing pointers to non-static members needs some std::bind (or std::mem_func),
// see http://stackoverflow.com/questions/37924781/. I prefer binding a seperate function.
template<int n_links, typename Pt = float3, Link_force<Pt> force = linear_force<Pt>>
void link_forces(Links<n_links>& links, const Pt* __restrict__ d_X, Pt* d_dX) {
    link<Pt, force><<<(links.get_d_n() + 32 - 1)/32, 32>>>(
        d_X, d_dX, links.d_link, links.get_d_n(), links.strength);
}
