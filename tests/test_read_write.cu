//1-Generates a solution object & writes down a vtk file
//2-Reads that vtk file and loads in into a solution object
//3-Compares the two solutions
//4-Writes down a 2nd vtk file with the values of the 2nd solution,
//  so that it can be compared with the 1st vtk file in Paraview

#include "../include/dtypes.cuh"
#include "../include/inits.cuh"
#include "../include/solvers.cuh"
#include "../include/vtk.cuh"
#include "../include/polarity.cuh"
#include "../include/property.cuh"
#include <string>

//#include "vtk_reader/vtk_reader.h"


const float min_diff=0.001f; //minimal difference allowed when comparing floats

const auto r_max=1.0;
const auto r_min=0.6;

const auto dt = 0.05*r_min*r_min;

const auto n_0 = 1000;
const auto n_max = 65000;

enum Cell_types {mesenchyme, epithelium};

//Properties for first solution (bolls)
__device__ Cell_types* d_type_o;
__device__ float* d_state_o;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

//Properties for second solution (balls)
__device__ Cell_types* d_type_a;
__device__ float* d_state_a;


MAKE_PT(Cell, x, y, z, theta, phi);


__device__ Cell relaxation_force(Cell Xi, Cell Xj, int i, int j) {
    Cell dF {0};

    if (i == j ) return dF;

    auto r = Xi - Xj;
    auto dist = norm3df(r.x, r.y, r.z);
    if (dist > r_max) return dF;

    auto F = 2*(r_min - dist)*(r_max - dist) + powf(r_max - dist, 2);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    if(d_type_o[i]==epithelium && d_type_o[j]==epithelium)
    {
      dF += rigidity_force(Xi, Xj)*0.2;//*3;
    }

    if (d_type_o[j] == epithelium) {atomicAdd(&d_epi_nbs[i],1);}
    else {atomicAdd(&d_mes_nbs[i],1);}

    return dF;
}

int main(int argc, char const *argv[]) {

    //1- Create a 1st solution (bolls) and write it down in a vtk file
    std::cout<<"1- Create a 1st solution (bolls) and write it down in a vtk file"<<std::endl;

    Solution<Cell, n_max, Lattice_solver> bolls(n_0);
    uniform_sphere(0.5, bolls);

    //int property cell type (actually is an enum)
    Property<n_max, Cell_types> type_o;
    cudaMemcpyToSymbol(d_type_o, &type_o.d_prop, sizeof(d_type_o));

    //float property state
    Property<n_max, float> state_o("state");
    cudaMemcpyToSymbol(d_state_o, &state_o.d_prop, sizeof(d_state_o));

    //initialise cell variables and properties
    for (auto i = 0; i < n_0; i++) {
        type_o.h_prop[i] = mesenchyme;
        state_o.h_prop[i] = rand()/(RAND_MAX+1.f);
    }
    bolls.copy_to_device();
    type_o.copy_to_device();
    state_o.copy_to_device();

    //these are just for simulation purposes, won't be written
    Property<n_max, int> n_mes_nbs;
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<n_max, int> n_epi_nbs;
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    // Relax the initial state
    for (auto time_step = 0; time_step <= 500; time_step++) {
        bolls.build_lattice(r_max);
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
        bolls.take_step<relaxation_force>(dt);
    }

    // Find epithelium
    bolls.copy_to_host();
    n_mes_nbs.copy_to_host();
    //printf("n_mes_nbs %i %i %i\n", n_mes_nbs.h_prop[0],n_mes_nbs.h_prop[1],n_mes_nbs.h_prop[2]);
    for (auto i = 0; i < n_0; i++) {
        if (n_mes_nbs.h_prop[i] < 35) {  // 2nd order solver
            type_o.h_prop[i] = epithelium;
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
    type_o.copy_to_device();

    // Relax again to let epithelium stabilise
    for (auto time_step = 0; time_step <= 600; time_step++) {
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
        bolls.take_step<relaxation_force>(dt);
    }

    // Integrate positions
    Vtk_output output_1("test_rw_1");
    bolls.copy_to_host();
    cudaDeviceSynchronize();
    output_1.write_positions(bolls);
    output_1.write_polarity(bolls);
    output_1.write_property(type_o);
    output_1.write_property(state_o);


    //2- Create a 2nd solution (balls) and load it with the file
    //   written before ("test_rw_1.vtk")
    std::cout<<"2- Create a 2nd solution (balls) and load it with the file written before (test_rw_1.vtk)"<<std::endl;

    //Declare properties
    Property<n_max, int> type_a;
    cudaMemcpyToSymbol(d_type_a, &type_a.d_prop, sizeof(d_type_a));

    Property<n_max, float> state_a("state");
    cudaMemcpyToSymbol(d_state_a, &state_a.d_prop, sizeof(d_state_a));

    std::string filename="output/test_rw_1_1.vtk";
    int n;
    Vtk_input input(filename,n);

    std::cout<<"system size= "<<n<<std::endl;

    Solution<Cell, n_max, Lattice_solver> balls(n);

    input.read_positions(balls);
    input.read_polarity(balls);
    input.read_property(type_a);
    input.read_property(state_a);

    //done reading
    //3- Compare the values stored in the 2 solutions
    std::cout<<"3- Compare the values stored in the 2 solutions"<<std::endl;

    Cell dif;
    for (int i=0 ; i<n ; i++)
    {
      //positions and polarities
      dif=bolls.h_X[i]-balls.h_X[i];
      if(dif.x>min_diff || dif.y>min_diff || dif.z>min_diff)
      {
        std::cout<<"ERROR in position "<<i<<" , difference is larger than "<<min_diff<<std::endl;
      }
      if(dif.theta>min_diff || dif.phi>min_diff)
      {
        std::cout<<"ERROR in polarity "<<i<<" , difference is larger than "<<min_diff<<std::endl;
      }
      //properties
      if(type_o.h_prop[i] != type_a.h_prop[i])
      {
        std::cout<<"ERROR in type "<<i<<" , it's different "<<std::endl;
      }
      if(state_o.h_prop[i] - state_a.h_prop[i] > min_diff)
      {
        std::cout<<"ERROR in state "<<i<<" , it's different "<<std::endl;
      }
    }

    //4-Writes down a 2nd vtk file with the values of the 2nd solution,
    //  so that it can be compared with the 1st vtk file in Paraview
    std::cout<<"4-Writes down a 2nd vtk file with the values of the 2nd solution, so that it can be compared with the 1st vtk file in Paraview"<<std::endl;

    Vtk_output output_2("test_rw_2");
    output_2.write_positions(balls);
    output_2.write_polarity(balls);
    output_2.write_property(type_a);
    output_2.write_property(state_a);

    return 0;
}
