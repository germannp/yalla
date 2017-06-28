//Simulation of limb bud growth starting with realistic limb bud shape
#include <curand_kernel.h>
#include "../../include/dtypes.cuh"
#include "../../include/inits.cuh"
#include "../../include/solvers.cuh"
#include "../../include/vtk.cuh"
#include "../../include/polarity.cuh"
#include "../../include/property.cuh"
#include "../../include/links.cuh"
#include <sstream>
#include <string>
#include <list>
#include <vector>
#include <iostream>

const auto r_max=1.0;
const auto r_min=0.8;
const auto dt = 0.05*r_min*r_min;
const auto n_max = 150000;
const auto const_proliferation_rate = 0.0004;

// enum Cell_types {mesenchyme, epithelium};

// __device__ Cell_types* d_type;
__device__ int* d_type;
__device__ int* d_mes_nbs;
__device__ int* d_epi_nbs;
// __device__ float* d_prolif_rate;

MAKE_PT(Cell, theta, phi);

__device__ Cell wall_force(Cell Xi, Cell r, float dist, int i, int j) {
    Cell dF {0};

    if(i==j) return dF;

    if (dist > r_max) return dF;

    float k;
    if(r_min-dist>0) //different coefficients for repulsion and adhesion
        k=8.f;
    else
        k=2.f;

    auto F = k*(r_min - dist);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    if(d_type[i]==1 && d_type[j]==1) dF += rigidity_force(Xi, r, dist)*0.15f;

    if(Xi.x<0) dF.x=0.f;

    if (d_type[j] == 1) {atomicAdd(&d_epi_nbs[i],1);}
    else {atomicAdd(&d_mes_nbs[i],1);}

    return dF;
}

__device__ float wall_friction(Cell Xi, Cell r, float dist, int i, int j) {
    if(Xi.x<0) return 0;
    return 1;
}

__global__ void proliferate(float mean_distance, Cell* d_X, int* d_n_cells, curandState* d_state) {
    D_ASSERT(*d_n_cells*const_proliferation_rate <= n_max);
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    //float proliferation_rate=d_prolif_rate[i];
    if (i >= *d_n_cells*(1 - const_proliferation_rate)) return;  // Dividing new cells is problematic!

    switch (d_type[i]) {
        case 0: {
            auto r = curand_uniform(&d_state[i]);
            if (r > const_proliferation_rate) return;
            break;
        }
        case 1: {
            // if (d_epi_nbs[i] > 10) return;
            auto r = curand_uniform(&d_state[i]);
            if (r > const_proliferation_rate) return;
            // return;
        }
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto phi = curand_uniform(&d_state[i])*M_PI;
    auto theta = curand_uniform(&d_state[i])*2*M_PI;
    d_X[n].x = d_X[i].x + mean_distance/4*sinf(theta)*cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance/4*sinf(theta)*sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance/4*cosf(theta);
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = 1;//d_type[i];
}


//*****************************************************************************

int main(int argc, char const *argv[]) {
    // Command line arguments
    // argv[1]=input file tag
    // argv[2]=output file tag
    // argv[3]=proliferation rate
    // argv[4]=time steps

    std::string file_name=argv[1];
    std::string output_tag=argv[2];

    //Load the initial conditions
    Vtk_input input(file_name);
    int n0=input.n_bolls;
    Solution<Cell, n_max, Grid_solver> limb(n0);

    Property<n_max, int> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));

    input.read_positions(limb);
    input.read_polarity(limb);
    input.read_property(type);

    limb.copy_to_device();
    type.copy_to_device();

    std::cout<<"initial nbolls "<<n0<<" nmax "<<n_max<<std::endl;

    Property<n_max, int> n_mes_nbs("n_mes_nbs");
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<n_max, int> n_epi_nbs("n_epi_nbs");
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    // State for proliferations
    curandState *d_state;
    cudaMalloc(&d_state, n_max*sizeof(curandState));
    setup_rand_states<<<(n_max + 128 - 1)/128, 128>>>(d_state, n_max);

    int n_time_steps=std::stoi(argv[4]);
    int skip_step=1;//n_time_steps/10;
    std::cout<<"n_time_steps "<<n_time_steps<<" write interval "<< skip_step<<std::endl;

    Vtk_output limb_output(output_tag);

    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        if(time_step%skip_step==0 || time_step==n_time_steps){
            limb.copy_to_host();
            n_epi_nbs.copy_to_host();
            n_mes_nbs.copy_to_host();
        }

        // std::cout<<"time step "<<time_step<<" d_n "<< limb.get_d_n()<<std::endl;

        proliferate<<<(limb.get_d_n() + 128 - 1)/128, 128>>>(r_min, limb.d_X, limb.d_n, d_state);

    // std::cout<<"after prolif "<<time_step<<" d_n "<< limb.get_d_n()<<std::endl;

        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + limb.get_d_n(), 0);
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + limb.get_d_n(), 0);
        // std::cout<<"after thrust_fill "<<time_step<<" d_n "<< limb.get_d_n()<<std::endl;

        limb.take_step<wall_force, wall_friction>(dt);

        // std::cout<<"after take step "<<time_step<<" d_n "<< limb.get_d_n()<<std::endl;

        //write the output
        if(time_step%skip_step==0 || time_step==n_time_steps) {
            limb_output.write_positions(limb);
            limb_output.write_polarity(limb);
            limb_output.write_property(type);
            limb_output.write_property(n_epi_nbs);
            limb_output.write_property(n_mes_nbs);
        }
    }

    return 0;
}
