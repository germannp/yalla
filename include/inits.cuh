// Initial states
#pragma once

#include <assert.h>
#include <time.h>
#include <iostream>


template<typename Pt, int n_max, template<typename, int> class Solver>
class Solution;


template<typename Pt, int n_max, template<typename, int> class Solver>
void random_disk(
    float dist_to_nb, Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
{
    assert(n_0 < *bolls.h_n);
    srand(time(NULL));
    // Radius based on hexagonal lattice
    auto r_max = pow((*bolls.h_n - n_0) / 0.9069, 1. / 2) * dist_to_nb / 2;
    for (auto i = n_0; i < *bolls.h_n; i++) {
        auto r = r_max * pow(rand() / (RAND_MAX + 1.), 1. / 2);
        auto phi = rand() / (RAND_MAX + 1.) * 2 * M_PI;
        bolls.h_X[i].x = 0;
        bolls.h_X[i].y = r * sin(phi);
        bolls.h_X[i].z = r * cos(phi);
    }
    bolls.copy_to_device();
}

template<typename Pt, int n_max, template<typename, int> class Solver>
void random_sphere(
    float dist_to_nb, Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
{
    assert(n_0 < *bolls.h_n);
    srand(time(NULL));
    // Radius based on random sphere packing
    auto r_max = pow((*bolls.h_n - n_0) / 0.64, 1. / 3) * dist_to_nb / 2;
    for (auto i = n_0; i < *bolls.h_n; i++) {
        auto r = r_max * pow(rand() / (RAND_MAX + 1.), 1. / 3);
        auto phi = rand() / (RAND_MAX + 1.) * 2 * M_PI;
        auto theta = acos(2. * rand() / (RAND_MAX + 1.) - 1);
        bolls.h_X[i].x = r * sin(theta) * cos(phi);
        bolls.h_X[i].y = r * sin(theta) * sin(phi);
        bolls.h_X[i].z = r * cos(theta);
    }
    bolls.copy_to_device();
}

template<typename Pt, int n_max, template<typename, int> class Solver>
void random_cuboid(float dist_to_nb, float3 mins, float3 dims,
    Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
{
    assert(n_0 < *bolls.h_n);

    auto cube_volume = dims.x * dims.y * dims.z;
    auto boll_volume = 4 / 3 * M_PI * pow(dist_to_nb / 2, 3);
    auto n = cube_volume / boll_volume * 0.64;  // Sphere packing

    assert(n_0 + n < n_max);
    *bolls.h_n = n_0 + n;

    srand(time(NULL));
    for (auto i = n_0; i < *bolls.h_n; i++) {
        bolls.h_X[i].x = mins.x + dims.x * (rand() / (RAND_MAX + 1.));
        bolls.h_X[i].y = mins.y + dims.y * (rand() / (RAND_MAX + 1.));
        bolls.h_X[i].z = mins.z + dims.z * (rand() / (RAND_MAX + 1.));
    }
    bolls.copy_to_device();
}


template<typename Pt>
__device__ Pt relu_force(Pt Xi, Pt r, float dist, int i, int j)
{
    Pt dF{0};

    if (i == j) return dF;

    if (dist > 1.f) return dF;

    auto F = fmaxf(0.8f - dist, 0) * 2.f - fmaxf(dist - 0.8f, 0);
    dF.x = r.x * F / dist;
    dF.y = r.y * F / dist;
    dF.z = r.z * F / dist;

    return dF;
}

template<typename Pt, int n_max, template<typename, int> class Solver>
void relaxed_sphere(
    float dist_to_nb, Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
{
    random_sphere(0.6, bolls, n_0);

    int relax_steps;
    if (*bolls.h_n <= 100) relax_steps = 500;
    else if (*bolls.h_n <= 1000) relax_steps = 1000;
    else if (*bolls.h_n <= 6000) relax_steps = 2000;
    else relax_steps = 3000;
    if (*bolls.h_n > 10000)
        std::cout << "Warning: The system is quite large, it may "
                  << "not be completely relaxed." << std::endl;

    for (int i = 0; i < relax_steps; i++)
        bolls.template take_step<relu_force>(0.1f);
    bolls.copy_to_host();
}

template<typename Pt, int n_max, template<typename, int> class Solver>
void relaxed_cuboid(float dist_to_nb, float3 mins, float3 dims,
    Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
{
    random_cuboid(0.8, mins, dims, bolls, n_0);

    int relax_steps;
    if (*bolls.h_n <= 3000) relax_steps = 1000;
    else if (*bolls.h_n <= 12000) relax_steps = 2000;
    else relax_steps = 3000;
    if (*bolls.h_n > 15000)
        std::cout << "Warning: The system is quite large, it may "
                  << "not be completely relaxed." << std::endl;

    for (int i = 0; i < relax_steps; i++)
        bolls.template take_step<relu_force>(0.1f);
    bolls.copy_to_host();
}


template<typename Pt, int n_max, template<typename, int> class Solver>
void regular_hexagon(
    float dist_to_nb, Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
{
    assert(n_0 < *bolls.h_n);

    auto beta = M_PI / 3.f;
    auto starting_angle = M_PI / 12.f;
    auto n_cells = *bolls.h_n;

    // Boll in center
    bolls.h_X[0].x = 0.f;
    bolls.h_X[0].y = 0.f;
    bolls.h_X[0].z = 0.f;
    auto cell_counter = 1;
    if (cell_counter >= n_cells) {
        bolls.copy_to_device();
        return;
    }
    auto i = 1;
    while (true) {
        for (auto j = 0; j < 6; j++) {
            // Main axis boll
            auto angle = starting_angle + beta * j;
            float3 p{-dist_to_nb * i * sinf(angle),
                dist_to_nb * i * cosf(angle), 0.f};
            bolls.h_X[cell_counter].x = p.x;
            bolls.h_X[cell_counter].y = p.y;
            bolls.h_X[cell_counter].z = p.z;
            cell_counter++;
            if (cell_counter >= n_cells) {
                bolls.copy_to_device();
                return;
            }
            // Intermediate bolls
            auto n_int = i - 1;
            if (n_int < 1) continue;
            auto next_angle = starting_angle + beta * (j + 1);
            float3 q{-dist_to_nb * i * sinf(next_angle),
                dist_to_nb * i * cosf(next_angle), 0.f};
            auto v = q - p;
            auto modulus = sqrt(pow(v.x, 2) + pow(v.y, 2));
            v = v * (1.f / modulus);
            for (auto k = 1; k <= n_int; k++) {
                auto u = v * modulus * (float(k) / float(n_int + 1));
                bolls.h_X[cell_counter].x = p.x + u.x;
                bolls.h_X[cell_counter].y = p.y + u.y;
                bolls.h_X[cell_counter].z = p.z + u.z;
                cell_counter++;
                if (cell_counter >= n_cells) {
                    bolls.copy_to_device();
                    return;
                }
            }
        }
        i++;
    }
}
