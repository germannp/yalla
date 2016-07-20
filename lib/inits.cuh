// Initial states
#include <assert.h>


template<typename Pt, int N_MAX, template<typename, int> class Solver>
class Solution;

// Distribute first n_cells in X[] uniformly random in circle
template<typename Pt, int N_MAX, template<typename, int> class Solver>
void uniform_circle(float mean_distance, Solution<Pt, N_MAX, Solver>& bolls,
        int n_cells = N_MAX) {
    auto r_max = pow(n_cells/0.9069, 1./2)*mean_distance/2;  // Circle packing
    for (auto i = 0; i < n_cells; i++) {
        auto r = r_max*pow(rand()/(RAND_MAX + 1.), 1./2);
        auto phi = rand()/(RAND_MAX + 1.)*2*M_PI;
        bolls.h_X[i].x = 0;
        bolls.h_X[i].y = r*sin(phi);
        bolls.h_X[i].z = r*cos(phi);
    }
    bolls.memcpyHostToDevice();
}

// Distribute first n_cells in X[] uniformly random in sphere
template<typename Pt, int N_MAX, template<typename, int> class Solver>
void uniform_sphere(float mean_distance, Solution<Pt, N_MAX, Solver>& bolls,
        int n_cells = N_MAX) {
    auto r_max = pow(n_cells/0.64, 1./3)*mean_distance/2;  // Sphere packing
    for (auto i = 0; i < n_cells; i++) {
        auto r = r_max*pow(rand()/(RAND_MAX + 1.), 1./3);
        auto theta = rand()/(RAND_MAX + 1.)*2*M_PI;
        auto phi = acos(2.*rand()/(RAND_MAX + 1.) - 1);
        bolls.h_X[i].x = r*sin(theta)*sin(phi);
        bolls.h_X[i].y = r*cos(theta)*sin(phi);
        bolls.h_X[i].z = r*cos(phi);
    }
    bolls.memcpyHostToDevice();
}
