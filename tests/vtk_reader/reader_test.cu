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


int main ()
{
  string filename="output/test_1.vtk";
  int n;
  Vtk_input input(filename,n);

  cout<<"system size= "<<n<<endl;

  Solution<Cell, n_max, Lattice_solver> bolls(n);

  input.read_positions(bolls);
  input.read_polarity(bolls);

  //done reading


  // Now we test that reading went right by writing it and comparing files

  Vtk_output output("reader_test");

  output.write_positions(bolls);
  output.write_polarity(bolls);

  return 0;

}
