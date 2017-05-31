//Simple program that generates different types of output the bolls vtk format
//writer class

#include "../../include/dtypes.cuh"
#include "../../include/inits.cuh"
#include "../../include/solvers.cuh"
#include "../../include/vtk.cuh"
#include "../../include/polarity.cuh"
#include "../../include/property.cuh"
#include "../../include/links.cuh"
#include <string>


const auto n_time_steps = 500;
const auto lambda = 1.;
//Turing parameters
//v1
// const auto D_u = 0.05;
// const auto D_v = 2.0;
// const auto f_v = 1.0;
// const auto f_u = 80.0;
// const auto g_u = 40.0;
// const auto m_u = 0.25;  //degradation rates
// const auto m_v = 0.5;  //
// const auto s_u = 0.05;
//v2
const auto D_u = 0.05;
const auto D_v = 2.0;
const auto f_v = 1.0;
const auto f_u = 80.0;
const auto g_u = 80.0;
const auto m_u = 0.25;  //degradation rates
const auto m_v = 0.75;  //
const auto s_u = 0.05;

const auto r_max=1.0;
const auto r_min=0.6;

const auto dt = 0.05*r_min*r_min/D_v;

const auto n_0 = 1000;
const auto n_max = 65000;

enum Cell_types {mesenchyme, epithelium};

__device__ Cell_types* d_type;
__device__ float* d_state;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;

MAKE_PT(Cell, x, y, z, theta, phi, u, v);


__device__ Cell relaxation_force(Cell Xi, Cell Xj, int i, int j) {
    Cell dF {0};

    // Turing reactions
    // Turing only takes place in epithelium
    if (i == j )
    {
      if(d_type[i]==epithelium)
      {
        //I'm trying with Meinhard equations
        dF.u = lambda*((f_u*Xi.u*Xi.u)/(1+f_v*Xi.v)-m_u*Xi.u + s_u);
        dF.v = lambda*(g_u*Xi.u*Xi.u - m_v*Xi.v);

        //in order to prevent negative values to arise
        if(-dF.u > Xi.u) dF.u=0.0f;
        if(-dF.v > Xi.v) dF.v=0.0f;
        //
      }
      return dF;
    }

    auto r = Xi - Xj;
    auto dist = norm3df(r.x, r.y, r.z);
    if (dist > r_max) return dF;

    auto F = 2*(r_min - dist)*(r_max - dist) + powf(r_max - dist, 2);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    //diffusion only takes place in epithelium
    if(d_type[i]==epithelium && d_type[j]==epithelium)
    {
      dF.u = -D_u*r.u;
      dF.v = -D_v*r.v;

      //in order to prevent negative values to arise
      if(-dF.u > Xi.u) dF.u=0.0f;
      if(-dF.v > Xi.v) dF.v=0.0f;
      //

      dF += rigidity_force(Xi, Xj)*0.2;//*3;

    }
    else
    {
      dF.v = -D_v*r.v; //inhibitor diffuses towards the mesenchyme, I'll use it to induce proliferation
    }

    if (d_type[j] == epithelium) {atomicAdd(&d_epi_nbs[i],1);}
    else {atomicAdd(&d_mes_nbs[i],1);}

    return dF;
}

int main(int argc, char const *argv[]) {

    Solution<Cell, n_max, Lattice_solver> bolls(n_0);
    uniform_sphere(0.5, bolls);
    //uniform_circle(0.5, bolls);
    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    for (auto i = 0; i < n_0; i++) {
        bolls.h_X[i].u = 0;
        bolls.h_X[i].v = 0;
        type.h_prop[i] = mesenchyme;
    }
    bolls.copy_to_device();
    type.copy_to_device();

    Property<n_max, float> state("state");
    cudaMemcpyToSymbol(d_state, &state.d_prop, sizeof(d_state));
    for (auto i = 0; i < n_0; i++) {
      state.h_prop[i] = rand()/(RAND_MAX+1.f);
    }
    state.copy_to_device();


    Property<n_max, int> n_mes_nbs;
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<n_max, int> n_epi_nbs;
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    // Relax
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
            type.h_prop[i] = epithelium;
            auto dist = sqrtf(bolls.h_X[i].x*bolls.h_X[i].x
                + bolls.h_X[i].y*bolls.h_X[i].y + bolls.h_X[i].z*bolls.h_X[i].z);
            bolls.h_X[i].theta = acosf(bolls.h_X[i].z/dist);
            bolls.h_X[i].phi = atan2(bolls.h_X[i].y, bolls.h_X[i].x);

            bolls.h_X[i].u = rand()/(RAND_MAX + 1.)/5 - 0.1;
            bolls.h_X[i].v = rand()/(RAND_MAX + 1.)/5 - 0.1;

        } else {
            bolls.h_X[i].theta = 0;
            bolls.h_X[i].phi = 0;
        }
    }
    bolls.copy_to_device();
    type.copy_to_device();

    // Relax again to let epithelium stabilise
    for (auto time_step = 0; time_step <= 100; time_step++) {
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
        bolls.take_step<relaxation_force>(dt);
    }

    // Integrate positions
    Vtk_output output("test");
    for (auto time_step = 0; time_step <= n_time_steps; time_step++) {
        bolls.copy_to_host();
        bolls.build_lattice(r_max);
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + bolls.get_d_n(), 0);
        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + bolls.get_d_n(), 0);

        bolls.take_step<relaxation_force>(dt);
        if(time_step==n_time_steps)
        {
          output.write_positions(bolls);
          output.write_polarity(bolls);
          output.write_property(type);
          output.write_property(state);
          //output.write_field(bolls, "u", &Cell::u);
          //output.write_field(bolls, "v", &Cell::v);
        }
    }

    return 0;
}
