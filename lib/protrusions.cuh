// Protrusions as constant forces between linked Pts
#include <cassert>
#include <curand_kernel.h>


template<int N_LINKS>
struct Protrusions {
    int links[N_LINKS][2];
    curandState rand_states[N_LINKS];
};


__global__ void setup_rand_states(curandState* state, int n_states) {
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_states) curand_init(1337, i, 0, &state[i]);
}

template<int N_LINKS>
void init_protrusions(Protrusions<N_LINKS>& prots) {
    for (auto i = 0; i < N_LINKS; i++) {
        prots.links[i][0] = 0;
        prots.links[i][1] = 0;
    }
    setup_rand_states<<<(N_LINKS + 32 - 1)/32, 32>>>(prots.rand_states, N_LINKS);
    cudaDeviceSynchronize();
}


template<typename Pt, int N_LINKS>
__global__ void intercalate(const Pt* __restrict__ X, Pt* dX, Protrusions<N_LINKS>& prots,
        float strength = 1.f/5, int n_links = N_LINKS) {
    assert(n_links <= N_LINKS);

    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_links) return;

    auto j = prots.links[i][0];
    auto k = prots.links[i][1];
    if (j == k) return;

    auto r = X[j] - X[k];
    auto dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    atomicAdd(&dX[j].x, -strength*r.x/dist);
    atomicAdd(&dX[j].y, -strength*r.y/dist);
    atomicAdd(&dX[j].z, -strength*r.z/dist);
    atomicAdd(&dX[k].x, strength*r.x/dist);
    atomicAdd(&dX[k].y, strength*r.y/dist);
    atomicAdd(&dX[k].z, strength*r.z/dist);
}
