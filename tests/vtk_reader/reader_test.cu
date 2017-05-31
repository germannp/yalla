//Program that reads from a bolls-generated vtk file and then writes that same
//data on a new vtk file using bolls vtk writer class

#include "../../include/dtypes.cuh"
#include "../../include/solvers.cuh"
#include "../../include/vtk.cuh"
#include "../../include/polarity.cuh"
#include "../../include/property.cuh"

#include "vtk_reader.h"

using namespace std;

MAKE_PT(Cell, x, y, z, theta, phi);

const auto n_max = 65000;

enum Cell_types {mesenchyme, epithelium};
__device__ Cell_types* d_type;

__device__ float* d_state;

//__device__ int* d_type;

int main ()
{

  //Initialise property
  Property<n_max, int> type;
  cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));

  Property<n_max, float> state("state");
  cudaMemcpyToSymbol(d_state, &state.d_prop, sizeof(d_state));


  string filename="output/test_1.vtk";
  int n;
  Vtk_input input(filename,n);

  cout<<"system size= "<<n<<endl;

  Solution<Cell, n_max, Lattice_solver> bolls(n);

  input.read_positions(bolls);
  input.read_polarity(bolls);
  // input.read_property_int(type);
  // input.read_property_float(state);
  input.read_property(type);
  input.read_property(state);


  //done reading


  // Now we test that reading went right by writing it and comparing files

  Vtk_output output("reader_test");

  output.write_positions(bolls);
  output.write_polarity(bolls);
  output.write_property(type);
  output.write_property(state);

  return 0;

}
