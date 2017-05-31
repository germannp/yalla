//Takes the mesenchyme tissue (as bolls) fitted from a limb 3D model (mesh)
//and makes limb bud initial conditions ready to feed into a model of limb morphogenesis.
//This implies:
//- adding an epithelial layer surrounding the mesenchyme.----DONE
//- making the tissue stable while being as close as possible to the original shape
//- Marking the boundary conditions (i.e. the surface that contacts the body flank)

//TODO
//- making the tissue stable while being as close as possible to the original shape
//      Set the initial polarities of epithelial cells normal to the surface
//      --> Load original mesh and take its normals.
//- Marking the boundary conditions (i.e. the surface that contacts the body flank)
//      Suggestion: the boundary surface is flat, all normals point in the same direction...


#include "../../include/dtypes.cuh"
#include "../../include/solvers.cuh"
#include "../../include/vtk.cuh"
#include "../../include/polarity.cuh"
#include "../../include/property.cuh"

using namespace std;

MAKE_PT(Cell, x, y, z, theta, phi);

const auto n_max = 65000;

const auto r_max=1.0;
const auto r_min=0.6;

const auto dt = 0.05*r_min*r_min;

enum Cell_types {mesenchyme, epithelium};

__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;
//__device__ float* d_r_mean;


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

    //diffusion only takes place in epithelium
    if(d_type[i]==epithelium && d_type[j]==epithelium)
    {
      dF += rigidity_force(Xi, Xj)*0.2;//*3;
    }

    if (d_type[j] == epithelium) {atomicAdd(&d_epi_nbs[i],1);}
    else {atomicAdd(&d_mes_nbs[i],1);}

    return dF;
}



int main (int argc, char const *argv[])
{

  //Command-line arguments
  //argv[1]: output file tag
  //argv[2]: input file path


  string filename=argv[2];
  int n;

  Vtk_input input(filename,n);

  cout<<"system size= "<<n<<endl;

  Solution<Cell, n_max, Lattice_solver> bolls(n);

  input.read_positions(bolls);
  input.read_polarity(bolls);

  Property<n_max, Cell_types> type;
  cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
  for (auto i = 0; i < n; i++) {
      type.h_prop[i] = mesenchyme;
  }
  bolls.copy_to_device();
  type.copy_to_device();
  Property<n_max, int> n_mes_nbs;
  cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
  Property<n_max, int> n_epi_nbs;
  cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));


  //done reading //now we'll add some cell types, epithelium on the surface, mesenchyme inside


  // we need to compute 1 iteration to compute the number of neighbours for each cell
  for (auto time_step = 0; time_step <= 1; time_step++) {
      bolls.build_lattice(r_max);

      // update_protrusions<<<(protrusions.get_d_n() + 32 - 1)/32, 32>>>(bolls.d_lattice,
      //     bolls.d_X, bolls.get_d_n(), protrusions.d_link, protrusions.d_state);
      thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n, 0);
      // bolls.take_step<lb_force>(dt, intercalation);

      bolls.take_step<relaxation_force>(dt);
  }

  // Find epithelium
  bolls.copy_to_host();
  n_mes_nbs.copy_to_host();

  for (auto i = 0; i < n; i++) {
      if (n_mes_nbs.h_prop[i] < 25) {
        type.h_prop[i] = epithelium;
        auto dist = sqrtf(bolls.h_X[i].x*bolls.h_X[i].x
            + bolls.h_X[i].y*bolls.h_X[i].y + bolls.h_X[i].z*bolls.h_X[i].z);
        bolls.h_X[i].theta = acosf(bolls.h_X[i].z/dist);
        bolls.h_X[i].phi = atan2(bolls.h_X[i].y, bolls.h_X[i].x);

      } else {
          bolls.h_X[i].theta = 0;
          bolls.h_X[i].phi = 0;
      }
      // bolls.h_X[i].w = 0;
      // bolls.h_X[i].f = 0;
  }
  bolls.copy_to_device();
  type.copy_to_device();

  // Relax again to let epithelium stabilise
  std::string output_tag=argv[1];
  std::string out_name=output_tag+"_limb_w_epi";
  Vtk_output output(out_name);
  for (auto time_step = 0; time_step <= 500; time_step++) {
      bolls.copy_to_host();
      bolls.build_lattice(r_max);

      // update_protrusions<<<(protrusions.get_d_n() + 32 - 1)/32, 32>>>(bolls.d_lattice,
      //     bolls.d_X, bolls.get_d_n(), protrusions.d_link, protrusions.d_state);
      thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n, 0);
      // bolls.take_step<lb_force>(dt, intercalation);

      bolls.take_step<relaxation_force>(dt);

      output.write_positions(bolls);
      output.write_polarity(bolls);
      output.write_property(type);

  }









  return 0;

}
