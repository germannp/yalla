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

enum Cell_types {mesenchyme, epithelium};

__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;
__device__ int* d_epi_nbs;
__device__ float* d_prolif_rate;

MAKE_PT(Cell, theta, phi);

__device__ Cell wall_force(Cell Xi, Cell r, float dist, int i, int j) {
    Cell dF {0};

    if(i==j) return dF;

    if (dist > r_max) return dF;

    // float k;
    // if(r_min-dist>0) { //different coefficients for repulsion and adhesion
    //     if(d_type[i] != d_type[j])
    //         k=8.0f;
    //     else
    //         k=8.f;
    // }else {
    //     if(d_type[i]==epithelium && d_type[j]==epithelium)
    //         k=8.f;
    //     else
    //         k=2.f;
    // }
    // auto F = k*(r_min - dist);
    float F;
    if (d_type[i] == d_type[j]) {
        if(d_type[i]==mesenchyme) F = fmaxf(0.8 - dist, 0)*8.f - fmaxf(dist - 0.8, 0)*2.f;
        else F = fmaxf(0.8 - dist, 0)*8.f - fmaxf(dist - 0.8, 0)*8.f;

    } else {
        F = fmaxf(0.9 - dist, 0)*8.f - fmaxf(dist - 0.9, 0)*2.f;
    }
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    if(d_type[i]==epithelium && d_type[j]==epithelium) dF += rigidity_force(Xi, r, dist)*0.5f;

    if(Xi.x<0) dF.x=0.f;

    if (d_type[j] == epithelium) {atomicAdd(&d_epi_nbs[i],1);}
    else {atomicAdd(&d_mes_nbs[i],1);}

    return dF;
}

__device__ float wall_friction(Cell Xi, Cell r, float dist, int i, int j) {
    if(Xi.x<0) return 0;
    return 1;
}

__global__ void proliferate(float max_rate, float mean_distance, Cell* d_X, int* d_n_cells,
        curandState* d_state) {
    D_ASSERT(*d_n_cells*max_rate <= n_max);
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= *d_n_cells*(1 - max_rate)) return;  // Dividing new cells is problematic!

    float rate=d_prolif_rate[i];

    switch (d_type[i]) {
        case mesenchyme: {
            auto r = curand_uniform(&d_state[i]);
            if (r > rate) return;
            break;
        }
        case epithelium: {
            if(d_epi_nbs[i]>15) return;
            if(d_mes_nbs[i]<=0) return;
            auto r = curand_uniform(&d_state[i]);
            if (r > 2.25f*rate) return;
        }
    }

    auto n = atomicAdd(d_n_cells, 1);
    auto theta = curand_uniform(&d_state[i])*2*M_PI;
    auto phi = curand_uniform(&d_state[i])*M_PI;
    d_X[n].x = d_X[i].x + mean_distance/4*sinf(theta)*cosf(phi);
    d_X[n].y = d_X[i].y + mean_distance/4*sinf(theta)*sinf(phi);
    d_X[n].z = d_X[i].z + mean_distance/4*cosf(theta);
    d_X[n].theta = d_X[i].theta;
    d_X[n].phi = d_X[i].phi;
    d_type[n] = d_type[i];
    d_prolif_rate[n] = d_prolif_rate[i];
    // d_mes_nbs[n] = 0;
    // d_epi_nbs[n] = 0;
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
    float max_proliferation_rate=std::stof(argv[3]);
    int n_time_steps=std::stoi(argv[4]);

    //Load the initial conditions
    Vtk_input input(file_name);
    int n0=input.n_bolls;
    Solution<Cell, n_max, Grid_solver> limb(n0);

    input.read_positions(limb);
    input.read_polarity(limb);

    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    Property<n_max, int> intype;

    input.read_property(intype); //we read it as an int, then we translate to enum "Cell_types"
    for(int i=0 ; i<n0 ; i++){
        if(intype.h_prop[i]==0) type.h_prop[i]=mesenchyme;
        else type.h_prop[i]=epithelium;
    }

    limb.copy_to_device();
    type.copy_to_device();

    std::cout<<"initial nbolls "<<n0<<" nmax "<<n_max<<std::endl;

    Property<n_max, int> n_mes_nbs("n_mes_nbs");
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<n_max, int> n_epi_nbs("n_epi_nbs");
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    //determine cell-specific proliferation rates
    Property<n_max, float> prolif_rate("prolif_rate");
    cudaMemcpyToSymbol(d_prolif_rate, &prolif_rate.d_prop, sizeof(d_prolif_rate));
    float xmax=-10000.0f;
    // float ymax=-10000.0f, ymin=10000.f;
    for(int i=0 ; i<n0 ; i++) {
        if(limb.h_X[i].x>xmax) xmax=limb.h_X[i].x;
        // if(limb.h_X[i].y>ymax) ymax=limb.h_X[i].y;
        // if(limb.h_X[i].y<ymin) ymin=limb.h_X[i].y;
    }
    for (int i=0; i<n0 ; i++) {
        if(limb.h_X[i].x<0) prolif_rate.h_prop[i]=0;
        else prolif_rate.h_prop[i]=pow((limb.h_X[i].x/xmax),3)*max_proliferation_rate;
    }
    // for (int i=0; i<n0 ; i++) {
    //     prolif_rate.h_prop[i]=pow((limb.h_X[i].y-ymin)/(ymax-ymin),2)*max_proliferation_rate;
    // }
    prolif_rate.copy_to_device();

    // State for proliferations
    curandState *d_state;
    cudaMalloc(&d_state, n_max*sizeof(curandState));
    setup_rand_states<<<(n_max + 128 - 1)/128, 128>>>(d_state, n_max);

    int skip_step=1;//n_time_steps/10;
    std::cout<<"n_time_steps "<<n_time_steps<<" write interval "<< skip_step<<std::endl;

    Vtk_output limb_output(output_tag);

    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        if(time_step%skip_step==0 || time_step==n_time_steps){
            limb.copy_to_host();
            type.copy_to_host();
            n_epi_nbs.copy_to_host();
            n_mes_nbs.copy_to_host();
            prolif_rate.copy_to_host();
        }

        proliferate<<<(limb.get_d_n() + 128 - 1)/128, 128>>>(max_proliferation_rate, r_min, limb.d_X, limb.d_n, d_state);

        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + limb.get_d_n(), 0);
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + limb.get_d_n(), 0);

        limb.take_step<wall_force, wall_friction>(dt);

        //write the output
        if(time_step%skip_step==0 || time_step==n_time_steps) {
            limb_output.write_positions(limb);
            limb_output.write_polarity(limb);
            limb_output.write_property(type);
            limb_output.write_property(n_epi_nbs);
            limb_output.write_property(n_mes_nbs);
            limb_output.write_property(prolif_rate);
        }
    }

    return 0;
}
