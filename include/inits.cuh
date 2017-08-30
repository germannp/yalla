// Initial states
#pragma once

#include <assert.h>
#include <time.h>


template<typename Pt, int n_max, template<typename, int> class Solver>
class Solution;

// Distribute bolls uniformly random in circle
template<typename Pt, int n_max, template<typename, int> class Solver>
void uniform_circle(float mean_distance, Solution<Pt, n_max, Solver>& bolls,
    unsigned int n_0 = 0)
{
    assert(n_0 < *bolls.h_n);
    srand(time(NULL));
    // Radius based on circle packing
    auto r_max = pow((*bolls.h_n - n_0) / 0.9069, 1. / 2) * mean_distance / 2;
    for (auto i = n_0; i < *bolls.h_n; i++) {
        auto r = r_max * pow(rand() / (RAND_MAX + 1.), 1. / 2);
        auto phi = rand() / (RAND_MAX + 1.) * 2 * M_PI;
        bolls.h_X[i].x = 0;
        bolls.h_X[i].y = r * sin(phi);
        bolls.h_X[i].z = r * cos(phi);
    }
    bolls.copy_to_device();
}

// Distribute bolls uniformly random in sphere
template<typename Pt, int n_max, template<typename, int> class Solver>
void uniform_sphere(float mean_distance, Solution<Pt, n_max, Solver>& bolls,
    unsigned int n_0 = 0)
{
    assert(n_0 < *bolls.h_n);
    srand(time(NULL));
    // Radius based on sphere packing
    auto r_max = pow((*bolls.h_n - n_0) / 0.64, 1. / 3) * mean_distance / 2;
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

// Distribute bolls uniformly random in cuboid
template<typename Pt, int n_max, template<typename, int> class Solver>
void uniform_cuboid(float xmin, float ymin, float zmin, float dx,
    float dy, float dz, Solution<Pt, n_max, Solver>& bolls,
    unsigned int n_0 = 0)
{
    assert(n_0 < *bolls.h_n);
    srand(time(NULL));
    for (auto i = n_0; i < *bolls.h_n; i++) {
        bolls.h_X[i].x = xmin + dx * (rand() / (RAND_MAX + 1.));
        bolls.h_X[i].y = ymin + dy * (rand() / (RAND_MAX + 1.));
        bolls.h_X[i].z = zmin + dz * (rand() / (RAND_MAX + 1.));
    }
    bolls.copy_to_device();
}
