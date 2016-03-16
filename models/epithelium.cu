// Simulating a layer.
#include <assert.h>
#include <cmath>
#include <iostream>
#include <stdio.h>

#include "../lib/inits.cuh"
#include "../lib/solvers.cuh"
#include "../lib/vtk.cuh"


const float R_MAX = 1;
const float R_MIN = 0.6;
const int N_CELLS = 20;
const int N_TIME_STEPS = 100;
const float DELTA_T = 0.5;


struct pocell {
    float x, y, z, phi, theta;
};

__device__ pocell operator+(const pocell& a, const pocell& b) {
    pocell sum = {a.x + b.x, a.y + b.y, a.z + b.z, a.phi + b.phi, a.theta + b.theta};
    return sum;
}

__device__ pocell operator*(const pocell& a, const float b) {
    pocell prod = {a.x*b, a.y*b, a.z*b, a.phi*b, a.theta*b};
    return prod;
}

__device__ __managed__ Solution<pocell, N_CELLS, LatticeSolver> X;


// Cubic potential plus (p_i . r_ij)^2/2r
__device__ pocell epithelium(pocell Xi, pocell Xj, int i, int j) {
    pocell dF = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (i == j) return dF;

    float3 r = {Xi.x - Xj.x, Xi.y - Xj.y, Xi.z - Xj.z};
    float dist = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (dist > R_MAX) return dF;

    float F = 2*(R_MIN - dist)*(R_MAX - dist) + powf(R_MAX - dist, 2);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    // n1 . n2 = sin(t1)*sin(t2)*cos(p1 - p2) + cos(t1)*cos(t2)
    float r_phi = atanf(r.y/r.x);
    float r_theta = acosf(r.z/dist);
    float prod = sinf(Xi.theta)*sinf(r_theta)*cosf(Xi.phi - r_phi) +
        cosf(Xi.theta)*cosf(r_theta);
    dF.phi = prod*(sinf(Xi.theta)*sinf(r_theta)*sinf(Xi.phi - r_phi));
    dF.theta = - prod*(cosf(Xi.theta)*sinf(r_theta)*cosf(Xi.phi - r_phi) -
        sinf(Xi.theta)*cosf(r_theta));

    assert(dF.x == dF.x);  // For NaN f != f.
    return dF;
}

__device__ __managed__ nhoodint<pocell> potential = epithelium;


// Write polarity
class PocellOutput: public VtkOutput {
public:
    using VtkOutput::VtkOutput;
    template<typename Pt, int N_MAX, template<typename, int> class Solver>
    void write_polarity(int n_cells, Solution<Pt, N_MAX, Solver>& X);
};

template<typename Pt, int N_MAX, template<typename, int> class Solver>
void PocellOutput::write_polarity(int n_cells, Solution<Pt, N_MAX, Solver>& X) {
    if (!mWrite) return;

    std::ofstream file(mCurrentFile, std::ios_base::app);
    assert(file.is_open());

    file << "\nPOINT_DATA " << n_cells << "\n";
    file << "NORMALS polarity float\n";
    float3 n = {0, 0, 0};
    for (int i = 0; i < n_cells; i++) {
        n.x = sinf(X[i].theta)*cosf(X[i].phi);
        n.y = sinf(X[i].theta)*sinf(X[i].phi);
        n.z = cosf(X[i].theta);
        file << n.x << " " << n.y << " " << n.z << "\n";
    }
}


int main(int argc, char const *argv[]) {
    // Prepare initial state
    // uniform_sphere(N_CELLS, 0.733333, X);
    uniform_circle(N_CELLS, 0.733333/2, X);
    for (int i = 0; i < N_CELLS; i++) {
        X[i].phi = rand()/(RAND_MAX + 1.)*M_PI;
        X[i].theta = rand()/(RAND_MAX + 1.)*2*M_PI;
    }

    // Integrate cell positions
    PocellOutput output("epithelium");
    for (int time_step = 0; time_step <= N_TIME_STEPS; time_step++) {
        output.write_positions(N_CELLS, X);
        output.write_polarity(N_CELLS, X);
        if (time_step == N_TIME_STEPS) return 0;

        X.step(DELTA_T, potential);
    }
}
