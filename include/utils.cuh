// Snippets used in several places
#pragma once

#include <curand_kernel.h>
#include <sstream>
#include <string>
#include <vector>


std::vector<std::string> split(const std::string& s)
{
    std::stringstream ss(s);
    std::string word;
    std::vector<std::string> words;
    while (std::getline(ss, word, ' ')) {
        words.push_back(word);
    }
    return words;
}


template<typename Pt_a, typename Pt_b>
__device__ __host__ float dot_product(Pt_a a, Pt_b b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


__global__ void setup_rand_states(int n_states, int seed, curandState* d_state)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_states) curand_init(seed, i, 0, &d_state[i]);
}
