// Snippets used in several places
#pragma once

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
