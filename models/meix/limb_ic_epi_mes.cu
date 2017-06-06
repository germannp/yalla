//Makes initial conditions for a limb bud taking the morphology from a 3D model
//(3D mesh), then fills the volume with mesenchymal cells and the surface with
//epithelial cells, then lets teh system relax.

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
#include <iostream>

#include "meix.h"

const auto r_max=1.0;
const auto r_min=0.6;

const auto dt = 0.05*r_min*r_min;

const auto n_0 = 1000;
const auto n_max = 65000;

enum Cell_types {mesenchyme, epithelium};

__device__ Cell_types* d_type;
 // __device__ int* d_mes_nbs;  // number of mesenchymal neighbours
 // __device__ int* d_epi_nbs;

//__device__ Cell_types* d_cube_type;


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

    auto F = 2*(r_min - dist)*(r_max - dist) + powf(r_max - dist, 2);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    if(d_type[i]==epithelium && d_type[j]==epithelium)
    {
      dF += rigidity_force(Xi, Xj)*0.2;//*3;
    }

    // if (d_type[j] == epithelium) {atomicAdd(&d_epi_nbs[i],1);}
    // else {atomicAdd(&d_mes_nbs[i],1);}

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

//this function will distribute uniformly random epithelial cells on top of the
//mesh surface.
void seed_epithelium_on_meix(Meix& meix, std::vector<Cell>& cells, float n_epi)
{

//std::cout<<"crash0.1"<<std::endl;
    int nF=meix.Facets.size();
    //std::cout<<"nF "<<nF<<std::endl;
    for(int i=0; i< n_epi; i++)
    {
      int j=rand()%nF; //which facet will fall into

      Point V0= meix.Facets[j].V0;
      Point V1= meix.Facets[j].V1;
      Point V2= meix.Facets[j].V2;
      Point N= meix.Facets[j].N;

      float phi=atan2(N.y,N.x);
      float theta= acos(N.z);

      bool bingo=false;
      while (!bingo)
      {
        // 0<s<1 ; 0<t<1 ; s+t<1
        float s=rand()/(RAND_MAX+1.f);
        float t=rand()/(RAND_MAX+1.f);
        if (s+t>1) continue;

        float a=1-s-t;

        Point p= V0*a + V1*s + V2*t;

        Cell c;

        c.x = p.x;
        c.y = p.y;
        c.z = p.z;
        c.phi = phi;
        c.theta = theta;
        cells.push_back(c);
//std::cout<<i<<" crash0.4 j= "<<j<<" c= "<<c<<" n= "<<n<<std::endl;
        bingo=true;
      }
    }
}

//*****************************************************************************

template<typename Pt, int n_max, template<typename, int> class Solver, typename Prop>
void epithelium_mesenchyme_assembly(std::vector<Point>& mes_cells, std::vector<Cell>& epi_cells, Property<n_max, Prop>& type, Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
{
  assert(n_0 < *bolls.h_n);
  int n_mes=mes_cells.size();
  int n_epi=epi_cells.size();

  for (int i=0 ; i<n_mes ; i++)
  {
    bolls.h_X[i].x = mes_cells[i].x;
    bolls.h_X[i].y = mes_cells[i].y;
    bolls.h_X[i].z = mes_cells[i].z;
    bolls.h_X[i].theta = 0.f;
    bolls.h_X[i].phi = 0.f;
    type.h_prop[i]=mesenchyme;
  }

  for (int i=0 ; i<n_epi ; i++)
  {
    bolls.h_X[n_mes+i].x = epi_cells[i].x;
    bolls.h_X[n_mes+i].y = epi_cells[i].y;
    bolls.h_X[n_mes+i].z = epi_cells[i].z;
    bolls.h_X[n_mes+i].theta = epi_cells[i].theta;
    bolls.h_X[n_mes+i].phi = epi_cells[i].phi;
    type.h_prop[i]=epithelium;
  }

  bolls.copy_to_device();
  type.copy_to_device();

}

//writes the whole meix data structure as a vtk file
void write_meix_vtk (Meix meix, std::string output_tag)
{
  std::string filename="output/"+output_tag+".meix.vtk";
  std::ofstream meix_file(filename);
  assert(meix_file.is_open());

  int n=meix.Facets.size();

  meix_file << "# vtk DataFile Version 3.0\n";
  meix_file << output_tag+".meix" << "\n";
  meix_file << "ASCII\n";
  meix_file << "DATASET POLYDATA\n";

  meix_file << "\nPOINTS " << 3*n << " float\n";
  for (auto i = 0; i < n; i++)
  {
    meix_file <<meix.Facets[i].V0.x << " " << meix.Facets[i].V0.y << " " << meix.Facets[i].V0.z << "\n";
    meix_file <<meix.Facets[i].V1.x << " " << meix.Facets[i].V1.y << " " << meix.Facets[i].V1.z << "\n";
    meix_file <<meix.Facets[i].V2.x << " " << meix.Facets[i].V2.y << " " << meix.Facets[i].V2.z << "\n";
  }

  meix_file << "\nPOLYGONS " << n << " " << 4*n << "\n";
  for (auto i = 0; i < 3*n; i+=3)
  {
    meix_file << "3 " << i <<" "<<i+1 <<" "<<i+2 << "\n";
  }
  meix_file.close();

}

int main(int argc, char const *argv[])
{

  // Command line arguments
  // argv[1]=output file tag
  // argv[2]=mesh file name
  // argv[3]=target limb bud size (dx)
  // argv[4]=cube relax_time
  // argv[5]=assumed cell radius
  // argv[6]=post assembly relax time

  std::string output_tag=argv[1];
  std::string file_name=argv[2];

  //First, load the mesh file so we can get the maximum dimensions of the system
  Meix meix(file_name);

  //Compute max length in X axis to know how much we need to rescale
  //**********************************************************************
  //Attention! we are assuming the PD axis of the limb is aligned with X
  //**********************************************************************

  float xmin=10000.0f,xmax=-10000.0f;
  float ymin,ymax,zmin,zmax;
  float dx,dy,dz;

  for(int i=0 ; i<meix.Facets.size() ; i++)
  {
    if(meix.Facets[i].C.x<xmin) xmin=meix.Facets[i].C.x;  if(meix.Facets[i].C.x>xmax) xmax=meix.Facets[i].C.x;
  }
  dx=xmax-xmin;

  float target_dx=std::stof(argv[3]);
  float resc=target_dx/dx;
  std::cout<<"xmax= "<<xmax<<" xmin= "<<xmin<<std::endl;
  std::cout<<"dx= "<<dx<<" target_dx= "<<target_dx<<" rescaling factor resc= "<<resc<<std::endl;

  meix.Rescale(resc);

  //Compute min. and max, positions in x,y,z from rescaled mesh
  xmin=10000.0f;xmax=-10000.0f;ymin=10000.0f;ymax=-10000.0f;zmin=10000.0f;zmax=-10000.0f;
  for(int i=0 ; i<meix.Facets.size() ; i++)
  {
    if(meix.Facets[i].C.x<xmin) xmin=meix.Facets[i].C.x;  if(meix.Facets[i].C.x>xmax) xmax=meix.Facets[i].C.x;
    if(meix.Facets[i].C.y<ymin) ymin=meix.Facets[i].C.y;  if(meix.Facets[i].C.y>ymax) ymax=meix.Facets[i].C.y;
    if(meix.Facets[i].C.z<zmin) zmin=meix.Facets[i].C.z;  if(meix.Facets[i].C.z>zmax) zmax=meix.Facets[i].C.z;
  }
  dx=xmax-xmin;
  dy=ymax-ymin;
  dz=zmax-zmin;

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

  std::cout<<"cube dims "<<dx<<" "<<dy<<" "<<dz<<std::endl;
  std::cout<<"nbolls in cube "<<n_bolls_cube<<std::endl;

  Solution<Cell, n_max, Lattice_solver> cube(n_bolls_cube);
  //Fill the rectangle with bolls
  uniform_cubic_rectangle(xmin,ymin,zmin,dx,dy,dz,cube);

  Property<n_max, Cell_types> type;
  cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
  for (auto i = 0; i < n_bolls_cube; i++)
  {
      type.h_prop[i] = mesenchyme;
  }

  cube.copy_to_device();
  type.copy_to_device();
  // Property<n_max, int> n_mes_nbs;
  // cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
  // Property<n_max, int> n_epi_nbs;
  // cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

  // We run the solver on bolls so the cube of bolls relaxes
  std::stringstream ass;
  ass << argv[1] << ".cubic_relaxation";
  std::string cubic_out = ass.str();

  int relax_time=std::stoi(argv[4]);
  int write_interval=relax_time/10;
  std::cout<<"relax_time "<<relax_time<<" write interval "<< write_interval<<std::endl;

  Vtk_output cubic_output(cubic_out);
  // std::cout<<"crash0"<<std::endl;
  for (auto time_step = 0; time_step <= relax_time; time_step++)
  {
    // std::cout<<"crash1"<<std::endl;
    if(time_step%write_interval==0 || time_step==relax_time)
    {
      // std::cout<<"crash2"<<std::endl;
      cube.copy_to_host();
    }
// std::cout<<"crash3"<<std::endl;
    cube.build_lattice(r_max);
// std::cout<<"crash4"<<std::endl;
    // thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_bolls_cube, 0);

    cube.take_step<relaxation_force>(dt);
// std::cout<<"crash5"<<std::endl;
    //write the output
    if(time_step%write_interval==0 || time_step==relax_time)
    {
      // std::cout<<"crash6"<<std::endl;
      cubic_output.write_positions(cube);
      cubic_output.write_polarity(cube);
      // std::cout<<"crash7"<<std::endl;
    }

  }


  //Find the bolls that are inside the mesh and store their positions
  //METHOD: Shooting a ray from a ball and counting how many triangles intersects.
  //If the ray intersects an even number of facets the boll is out of the mesh, else is in

  //Setup the list of points
  std::vector<Point> points;
  for (auto i = 0; i < n_bolls_cube; i++)
  {
    Point p=Point(cube.h_X[i].x, cube.h_X[i].y, cube.h_X[i].z);
    points.push_back(p);
  }

  //Setup the list of inclusion test results
  int* results=new int[n_bolls_cube];
  //Set direction of ray
  Point dir=Point(1.0f,0.0f,0.0f);

  meix.InclusionTest(points , results, dir);

  //Make a new list with the ones that are inside
  std::vector<Point> mes_cells;
  int n_bolls_mes=0;
  for (int i = 0; i < n_bolls_cube; i++)
  {
    if(results[i]==1)
    {
      mes_cells.push_back(points[i]);
      n_bolls_mes++;
    }
  }

  std::cout<<"bolls_in_cube "<<n_bolls_cube<<" bolls after fill "<<n_bolls_mes<<std::endl;


  //The mesenchyme is done, now we make the epithelium

  //In order to efficiently seed the mesh surface, we need to estimate epithelial
  //cell density based on the cell size. We have cell radius (bolls eq. radius)
  //as input from cmd. line. We assume hexagonal packing of epithelia, so the
  //effective surface occupied by one cell will be the one of an hexagon with
  //apothem equal to the cell radius

  float cell_radius=std::stof(argv[5]);
  float cell_S=cell_radius*cell_radius*6.f/sqrt(3.f); //regular hexagon formula
  float cell_density=1.f/cell_S;

  //Calculate whole Surface area of meix
  int n_bolls_epi=std::round(cell_density*meix.SurfArea);

  std::cout<<"nbolls_epi= "<<n_bolls_epi<<" cell_density= "<<cell_density<<" meix_S= "<<meix.SurfArea<<std::endl;

  std::vector<Cell> epi_cells;

  //seed the cells onto the meix
  seed_epithelium_on_meix(meix, epi_cells,n_bolls_epi);

  int n_bolls_total=n_bolls_mes+n_bolls_epi;
  Solution<Cell, n_max, Lattice_solver> bolls(n_bolls_total);
  // Property<n_max, Cell_types> type;
  // cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));

  epithelium_mesenchyme_assembly(mes_cells, epi_cells, type,bolls);

  // for (int i = 0; i < n_bolls; i++)
  // {
  //     type.h_prop[i] = epithelium;
  // }
  // bolls.copy_to_device();
  // type.copy_to_device();

  // Property<n_max, int> n_mes_nbs;
  // cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
  // Property<n_max, int> n_epi_nbs;
  // cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));


Vtk_output output(output_tag);

relax_time=std::stoi(argv[6]);
write_interval=relax_time/10;

for (auto time_step = 0; time_step <= relax_time; time_step++)
{
  if(time_step%write_interval==0 || time_step==relax_time)
  {
    bolls.copy_to_host();
  }

  bolls.build_lattice(r_max);

  //thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);

  bolls.take_step<relaxation_force>(dt);

  //write the output
  if(time_step%write_interval==0 || time_step==relax_time)
  {
    output.write_positions(bolls);
    output.write_polarity(bolls);
    output.write_property(type);
  }

}

  //write down the meix in the vtk file to compare it with the posterior seeding
  write_meix_vtk(meix,output_tag);

  return 0;
}
