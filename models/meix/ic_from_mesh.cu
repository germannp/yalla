//Create the initial conditions for bolls with a surface mesh file (STL)

#include "../../include/dtypes.cuh"
#include "../../include/inits.cuh"
#include "../../include/solvers.cuh"
#include "../../include/vtk.cuh"
#include "../../include/polarity.cuh"
#include "../../include/property.cuh"
#include <sstream>
#include <string>
#include <list>
#include <vector>

#include "meix.h"
#include "meix_inclusion_test.h"

const auto r_max=1.0;
const auto r_min=0.6;

const auto dt = 0.05*r_min*r_min;

const auto n_0 = 1000;
const auto n_max = 65000;

enum Cell_types {mesenchyme, epithelium};

__device__ Cell_types* d_type;
__device__ int* d_mes_nbs;  // number of mesenchymal neighbours
__device__ int* d_epi_nbs;


MAKE_PT(Cell, x, y, z, theta, phi);

__device__ Cell relaxation_force(Cell Xi, Cell Xj, int i, int j) {
    Cell dF {0};

    if(i==j) return dF;

    //FOR TESTING PURPOSES ONLY //we use epithelium type to reproduce the mesh within the bolls framework
    if(d_type[i]==epithelium || d_type[j]==epithelium) return dF;
    //*************************//

    auto r = Xi - Xj;
    auto dist = norm3df(r.x, r.y, r.z);
    if (dist > r_max) return dF;

    auto F = 2*(r_min - dist)*(r_max - dist) + powf(r_max - dist, 2);
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

// Distribute bolls uniformly random in rectangular cube
template<typename Pt, int n_max, template<typename, int> class Solver>
void uniform_cubic_rectangle(float x0,float y0,float z0,float dx,float dy,float dz, Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
{

    assert(n_0 < *bolls.h_n);

    for (auto i = n_0; i < *bolls.h_n; i++) {
      //std::cout<<"crash5.1.3 "<< i<<std::endl;
        bolls.h_X[i].x = x0+dx*(rand()/(RAND_MAX+1.));
        bolls.h_X[i].y = y0+dy*(rand()/(RAND_MAX+1.));
        bolls.h_X[i].z = z0+dz*(rand()/(RAND_MAX+1.));
        bolls.h_X[i].phi=0.0f;
        bolls.h_X[i].theta=0.0f;

    }

    bolls.copy_to_device();
}

//In order to visualise the mesh once loaded to the bolls framework
//we create an epithelial boll on the centre of each facet of the mesh
//(these bolls go into a different solution object)
template<typename Pt, int n_max, template<typename, int> class Solver>
void fill_solver_from_meix(std::vector<Facet> F_list, Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
{

    assert(n_0 < *bolls.h_n);

    Facet f;
    int i=0;
    //for(std::vector<Facet>::iterator fit = F_list.begin(); fit != F_list.end(); fit++)
    //for(int fit = 0 ; fit <= 10 ; fit++)
    for(int fit = 0 ; fit <= F_list.size() ; fit++)
    {
      //f=*fit;
      f=F_list[fit];
      //std::cout<<"cxyz "<<f.cx<<" "<<f.cy<<" "<<f.cz<<std::endl;
      float r=sqrt(pow(f.nx,2)+pow(f.ny,2)+pow(f.nz,2));
      bolls.h_X[i].x = f.cx;
      bolls.h_X[i].y = f.cy;
      bolls.h_X[i].z = f.cz;
      bolls.h_X[i].phi = atan2(f.ny,f.nx);
      bolls.h_X[i].theta = acos(f.nz/r);
      i++;
    }

    bolls.copy_to_device();
}


// Insert cells into the solver from an array of 3D points (custom class Point, see meix_inclusion_test.h)
template<typename Pt, int n_max, template<typename, int> class Solver>
void fill_solver_from_list(std::vector<Point> points, Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
{
    assert(n_0 < *bolls.h_n);

    for (auto i = n_0; i < *bolls.h_n; i++) {
        bolls.h_X[i].x = points[i].x;
        bolls.h_X[i].y = points[i].y;
        bolls.h_X[i].z = points[i].z;
    }

    bolls.copy_to_device();
}

int main(int argc, char const *argv[])
{

  //Here's the plan:
  //1. We fill a space larger than the object we want to fill (the mesh) with bolls
  //2. We delete the bolls that have fallen outside the mesh (using the normals of the facets, or triangles)


  // Command line arguments
  // argv[1]=output file name
  // argv[2]=mesh file name
  // argv[3]=mesh rescaling factor (meshes that are too large take a crazy amount of bolls)
  // argv[4]= # time steps for relaxation stage (make sure the system is mechanically stable)

  std::string file_name=argv[2];

  //First, load the mesh file so we can get the maximum dimensions of the system
  std::vector<Facet> facets0; //store original mesh
  std::vector<Facet> facets;  //store rescaled mesh
  GetMeixSTL(file_name,facets0);

  //we need to rescale because it's too large
  //std::vector<Facet> facets;
  //float resc=0.05; //sphere
  //float resc=0.02; //limb10.04
  //float resc=0.01; //limb12.04
  float resc=stof(argv[3]);

  MeixRescale(facets0, facets, resc);

  //Compute min. and max, positions in x,y,z from mesh file

  float xmin=10000.0f,xmax=-10000.0f,ymin=100000.0f,ymax=-10000.0f,zmin=1000000.0f,zmax=-10000.0f;

  for(std::vector<Facet>::iterator fit = facets.begin(); fit != facets.end(); fit++)
  {
    Facet f=*fit;

    if(f.cx<xmin) xmin=f.cx;  if(f.cx>xmax) xmax=f.cx;
    if(f.cy<ymin) ymin=f.cy;  if(f.cy>ymax) ymax=f.cy;
    if(f.cz<zmin) zmin=f.cz;  if(f.cz>zmax) zmax=f.cz;
  }
  float dx=xmax-xmin;
  float dy=ymax-ymin;
  float dz=zmax-zmin;

  //Translocation and rotation of the mesh, if convenient?
  //TODO
  //

  //we use the maximum lengths of the mesh to draw a cube that includes the mesh
  //Let's fill the cube with bolls
  //How many bolls? We calculate the volume of the cube we want to fill
  //then we calculate how many bolls add up to that volume, correcting by the
  //inefficiency of a cubic packing (0.74)----> Well in the end we don't correct cause it wasn't packed enough

  //const float packing_factor=0.74048f;
  float cube_vol=dx*dy*dz;
  float r_boll=0.3f;
  float boll_vol=4./3.*M_PI*pow(r_boll,3);
  int n_bolls_cube=cube_vol/boll_vol;

  std::cout<<"dims "<<dx<<" "<<dy<<" "<<dz<<std::endl;
  std::cout<<"nbolls in cube "<<n_bolls_cube<<std::endl;

  Solution<Cell, n_max, Lattice_solver> bolls(n_bolls_cube);
  //Fill the rectangle with bolls
  uniform_cubic_rectangle(xmin,ymin,zmin,dx,dy,dz,bolls);

  Property<n_max, Cell_types> type;
  cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
  for (auto i = 0; i < n_bolls_cube; i++)
  {
      type.h_prop[i] = mesenchyme;
  }

  bolls.copy_to_device();
  type.copy_to_device();
  Property<n_max, int> n_mes_nbs;
  cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
  Property<n_max, int> n_epi_nbs;
  cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

  // We run the solver on bolls so the cube of bolls relaxes
  std::stringstream ass;
  ass << argv[1] << ".cubic_relaxation";
  std::string cubic_out = ass.str();

  int relax_time=stoi(argv[4]);
  int write_interval=relax_time/10;
  std::cout<<"relax_time "<<relax_time<<" write interval "<< write_interval<<std::endl;

  Vtk_output output(cubic_out);

  for (auto time_step = 0; time_step <= relax_time; time_step++)
  {
    if(time_step%write_interval==0 || time_step==relax_time)
    {
      bolls.copy_to_host();
    }

    bolls.build_lattice(r_max);

    thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);

    bolls.take_step<relaxation_force>(dt);

    //write the output
    if(time_step%write_interval==0 || time_step==relax_time)
    {
      output.write_positions(bolls);
      output.write_polarity(bolls);
    }

  }

  //Find the bolls that are inside the mesh and store their positions
  //METHOD: Shooting a ray from a ball and counting how many triangles intersects.
  //If the ray intersects an even number of facets the boll is out of the mesh, else is in

  //Setup the list of points
  std::vector<Point> points;
  for (auto i = 0; i < n_bolls_cube; i++)
  {
    Point p=Point(bolls.h_X[i].x, bolls.h_X[i].y, bolls.h_X[i].z);
    points.push_back(p);
  }

  //Setup the list of inclusion test results
  int* results=new int[n_bolls_cube];
  //Set direction of ray
  Point dir=Point(1.0f,0.0f,0.0f);

  MeixInclusionTest(points , facets , results, dir);

  //Make a new list with the ones that are inside
  std::vector<Point> points_fit;
  int n_bolls_fit=0;
  for (int i = 0; i < n_bolls_cube; i++)
  {
    if(results[i]==1)
    {
      points_fit.push_back(points[i]);
      n_bolls_fit++;
    }
  }

  std::cout<<"bolls_in_cube "<<n_bolls_cube<<" bolls after fit "<<n_bolls_fit<<std::endl;

  Solution<Cell, n_max, Lattice_solver> balls(n_bolls_fit);
  fill_solver_from_list(points_fit,balls);

  //Write fitted morphology on output file
  Vtk_output output_2(argv[1]);
  balls.copy_to_host();
  output_2.write_positions(balls);
  output_2.write_polarity(balls);

  //Relax the fitted morphology?
  // for (auto time_step = 0; time_step <= 1; time_step++)
  // {
  //     balls.copy_to_host();
  //
  //     balls.build_lattice(r_max);
  //
  //     // update_protrusions<<<(protrusions.get_d_n() + 32 - 1)/32, 32>>>(bolls.d_lattice,
  //     //     bolls.d_X, bolls.get_d_n(), protrusions.d_link, protrusions.d_state);
  //     thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
  //     // bolls.take_step<lb_force>(dt, intercalation);
  //
  //     balls.take_step<relaxation_force>(dt);
  //
  //     //write the output
  //
  //     output_2.write_positions(balls);
  //     output_2.write_polarity(balls);
  //
  // }

  //For testing purposes, I want to see the mesh along with the cells, so I will put an epithelial cell at the centre
  //of every facet, just to see where the limits of the mesh are.
  Solution<Cell, n_max, Lattice_solver> meix(facets.size());

  fill_solver_from_meix(facets,meix);

  std::stringstream ss;
  ss << argv[1] << ".meix" ;
  std::string meix_out = ss.str();

  Vtk_output output_meix(meix_out);
  output_meix.write_positions(meix);
  output_meix.write_polarity(meix);
  //std::cout<<"meix size: number of facets: "<<facets.size()<<std::endl;
  //****************************************************************//

  return 0;
}
