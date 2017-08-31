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
