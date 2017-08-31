// Initial states
#pragma once

#include <assert.h>
#include <time.h>
#include <iostream>

template<typename Pt>
__device__ Pt relaxation_linear_force(Pt Xi, Pt r, float dist, int i, int j)
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

template<typename Pt>
__device__ float local_friction(Pt Xi, Pt r, float dist, int i, int j)
{
    if(i == j) return 0;
    return 1;
}

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

    //relax initial state
    int relax_time;
    if(*bolls.h_n <= 100) relax_time = 1000;
    else if(*bolls.h_n <= 1000) relax_time = 2000;
    else if(*bolls.h_n <= 1500) relax_time = 3000;
    else relax_time = 4000;
    if(*bolls.h_n > 2000)
        std::cout<<"The system is quite large, most likely"<<
            " is not going to be mechanically relaxed"<<std::endl;

    for (int i = 0 ; i < relax_time ; i++)
        bolls. template take_step<relaxation_linear_force,
            local_friction>(0.1f);
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

    //relax initial state
    int relax_time;
    if(*bolls.h_n <= 100) relax_time = 500;
    else if(*bolls.h_n <= 1000) relax_time = 1000;
    else if(*bolls.h_n <= 6000) relax_time = 2000;
    else relax_time = 3000;
    if(*bolls.h_n > 10000)
        std::cout<<"The system is quite large, most likely"<<
            " is not going to be mechanically relaxed"<<std::endl;

    for (int i = 0 ; i < relax_time ; i++)
        bolls. template take_step<relaxation_linear_force,
            local_friction>(0.1f);
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

// Distribute bolls with regular hexagonal distribution
template<typename Pt, int n_max, template<typename, int> class Solver>
void regular_hexagon(float mean_distance, Solution<Pt, n_max, Solver>& bolls,
    unsigned int n_0 = 0)
{
    assert(n_0 < *bolls.h_n);

    auto beta = M_PI / 3.f;
    auto starting_angle = M_PI / 12.f;
    auto n_cells = *bolls.h_n;

    //boll in the centre;
    bolls.h_X[0].x = 0.f;
    bolls.h_X[0].y = 0.f;
    bolls.h_X[0].z = 0.f;
    auto cell_counter = 1;
    if(cell_counter >= n_cells) {
        bolls.copy_to_device();
        return;
    }
    auto i = 1;
    while (true) {
        for (auto j = 0 ; j < 6 ; j++) {
            //main axis node
            auto angle = starting_angle + beta * j;
            float3 p {-mean_distance * i * sinf(angle),
                mean_distance * i * cosf(angle), 0.f};
            bolls.h_X[cell_counter].x = p.x;
            bolls.h_X[cell_counter].y = p.y;
            bolls.h_X[cell_counter].z = p.z;
            cell_counter++;
            if(cell_counter >= n_cells) {
                bolls.copy_to_device();
                return;
            }
            //intermediate nodes
            auto n_int = i - 1;
            if(n_int < 1) continue;
            auto next_angle = starting_angle + beta * (j + 1);
            float3 q {-mean_distance * i * sinf(next_angle),
                mean_distance * i * cosf(next_angle), 0.f};
            auto v = q - p;
            auto modulus = sqrt(pow(v.x, 2) + pow(v.y, 2));
            v = v * (1.f / modulus);
            for (auto k = 1 ; k <= n_int ; k++) {
                auto u = v * modulus * (float(k) / float(n_int + 1));
                bolls.h_X[cell_counter].x = p.x + u.x;
                bolls.h_X[cell_counter].y = p.y + u.y;
                bolls.h_X[cell_counter].z = p.z + u.z;
                cell_counter++;
                if(cell_counter >= n_cells) {
                    bolls.copy_to_device();
                    return;
                }
            }
        }
        i++;
    }

}
