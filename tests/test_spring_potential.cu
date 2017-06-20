// This program simulates a branching mechanism on a spheric organoid.
// A Turing mechanism taking place on its surface creates a pattern,
// peaks of activator induce local proliferation on the underlying cells,
// resulting on the growth of a branch.

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include <curand_kernel.h>
#include "../include/links.cuh"
#include <string>


const auto r_max = 1.0;
const auto r_min = 0.8;
int write_interval=100;
const auto dt = 0.05*r_min*r_min;

const auto n_0 = 1000;
const auto n_max = 65000;

enum Cell_types {mesenchyme, epithelium};

__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;
//__device__ float* d_r_mean;

MAKE_PT(Cell, x, y, z, theta, phi);


__device__ Cell relaxation_force(Cell Xi, Cell Xj, int i, int j) {
    Cell dF {0};

    if(i==j) return dF;

    //FOR TESTING PURPOSES ONLY //we use epithelium type to reproduce the mesh within the bolls framework
    //if(d_type[i]==epithelium || d_type[j]==epithelium) return dF;
    //*************************//

    auto r = Xi - Xj;
    auto dist = norm3df(r.x, r.y, r.z);
    if (dist > r_max) return dF;

    //auto F = 2*(r_min - dist)*(r_max - dist) + powf(r_max - dist, 2);
    auto F = 2.f*(r_min - dist);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    if(d_type[i]==epithelium && d_type[j]==epithelium)
    {
      dF += rigidity_force(Xi, Xj)*0.2;//*3;
    }

     if (d_type[j] == epithelium) {atomicAdd(&d_epi_nbs[i],1);}
     else {atomicAdd(&d_mes_nbs[i],1);}

    return dF;
}

int main(int argc, char const *argv[]) {

    //Command line arguments
    //argv[1] : output file taking
    //argv[2] : number of time steps

    //New initial state, a sphere of balls (with epithelium on the surface
    //and mesenchyme inside (I'm just copying and adapting the code from elongation.cu)

    auto n_time_steps = std::stoi(argv[2]);

    Solution<Cell, n_max, Grid_solver> bolls(n_0);
    uniform_sphere(0.5, bolls);
    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (auto i = 0; i < n_0; i++) {
        type.h_prop[i] = mesenchyme;
    }
    bolls.copy_to_device();
    type.copy_to_device();
    Property<n_max, int> n_mes_nbs;
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<n_max, int> n_epi_nbs;
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    // Relax
    Vtk_output output(argv[1]);
    for (auto time_step = 0; time_step <= 1000; time_step++) {
        bolls.copy_to_host();

        bolls.build_grid(r_max);

        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);

        bolls.take_step<relaxation_force>(dt);

        output.write_positions(bolls);
        output.write_polarity(bolls);
    }

    // Find epithelium
    bolls.copy_to_host();
    n_mes_nbs.copy_to_host();
    //printf("n_mes_nbs %i %i %i\n", n_mes_nbs.h_prop[0],n_mes_nbs.h_prop[1],n_mes_nbs.h_prop[2]);
    for (auto i = 0; i < n_0; i++) {
        if (n_mes_nbs.h_prop[i] < 35) {
            type.h_prop[i] = epithelium;
            auto dist = sqrtf(bolls.h_X[i].x*bolls.h_X[i].x
                + bolls.h_X[i].y*bolls.h_X[i].y + bolls.h_X[i].z*bolls.h_X[i].z);
            bolls.h_X[i].theta = acosf(bolls.h_X[i].z/dist);
            bolls.h_X[i].phi = atan2(bolls.h_X[i].y, bolls.h_X[i].x);

        } else {
            bolls.h_X[i].theta = 0;
            bolls.h_X[i].phi = 0;
        }

    }
    bolls.copy_to_device();
    type.copy_to_device();

    // Integrate positions
    // Vtk_output output("branching");
    //Vtk_output output(argv[1]);
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();

        // type.copy_to_device();
        // printf("type %d %d %d\n", type.h_prop[0],type.h_prop[1],type.h_prop[2]);
        //
        // n_epi_nbs.copy_to_host();
        // printf("n_epi_nbs %i %i %i\n", n_epi_nbs.h_prop[0],n_epi_nbs.h_prop[1],n_epi_nbs.h_prop[2]);
        //
        // n_mes_nbs.copy_to_host();
        // printf("n_mes_nbs %i %i %i\n", n_mes_nbs.h_prop[0],n_mes_nbs.h_prop[1],n_mes_nbs.h_prop[2]);
        //
        // r_mean.copy_to_host();
        // printf("r_mean %f %f %f\n", r_mean.h_prop[0],r_mean.h_prop[1],r_mean.h_prop[2]);
        //
        // printf("\n****TIME STEP******** %d\n",time_step);

        bolls.build_grid(r_max);
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + bolls.get_d_n(), 0);

        //thrust::fill(thrust::device, r_mean.d_prop, r_mean.d_prop + bolls.get_d_n(), 0);

        bolls.take_step<relaxation_force>(dt);
        // if(time_step%write_interval==0)
        // {
          //auto n_cells=bolls.get_d_n();
          //printf("time_step %i n after prolif %i\n",time_step,n_cells);

          output.write_positions(bolls);
          output.write_polarity(bolls);
        // }
    }

    return 0;
}
