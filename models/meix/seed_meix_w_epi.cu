//Test program that seeds the surface of a mesh with uniformly random distributed
//epithelial cells

//The skeleton (command line arguments and most of the code) is the same as ic_from_meix.cu

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
//#include "meix_inclusion_test.h"

const auto r_max=1.5;
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

    if (d_type[j] == epithelium) {atomicAdd(&d_epi_nbs[i],1);}
    else {atomicAdd(&d_mes_nbs[i],1);}

    return dF;
}


//In order to visualise the mesh once loaded to the bolls framework
//we create an epithelial boll on the centre of each facet of the mesh
//(these bolls go into a different solution object)
// template<typename Pt, int n_max, template<typename, int> class Solver>
// void fill_solver_from_meix(std::vector<Facet> F_list, Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
// {
//
//     assert(n_0 < *bolls.h_n);
//
//     Facet f;
//     int i=0;
//     //for(std::vector<Facet>::iterator fit = F_list.begin(); fit != F_list.end(); fit++)
//     //for(int fit = 0 ; fit <= 10 ; fit++)
//     for(int fit = 0 ; fit <= F_list.size() ; fit++)
//     {
//       //f=*fit;
//       f=F_list[fit];
//       //std::cout<<"cxyz "<<f.cx<<" "<<f.cy<<" "<<f.cz<<std::endl;
//       float r=sqrt(pow(f.nx,2)+pow(f.ny,2)+pow(f.nz,2));
//       bolls.h_X[i].x = f.cx;
//       bolls.h_X[i].y = f.cy;
//       bolls.h_X[i].z = f.cz;
//       bolls.h_X[i].phi = atan2(f.ny,f.nx);
//       bolls.h_X[i].theta = acos(f.nz/r);
//       i++;
//     }
//
//     bolls.copy_to_device();
// }

// // Insert cells into the solver from an array of 3D points (custom class Point, see meix_inclusion_test.h)
// template<typename Pt, int n_max, template<typename, int> class Solver>
// void fill_solver_from_list(std::vector<Point> points, Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0)
// {
//     assert(n_0 < *bolls.h_n);
//
//     for (auto i = n_0; i < *bolls.h_n; i++) {
//         bolls.h_X[i].x = points[i].x;
//         bolls.h_X[i].y = points[i].y;
//         bolls.h_X[i].z = points[i].z;
//     }
//
//     bolls.copy_to_device();
// }
//
float meix_get_surfarea(std::vector<Triangle>& F, float cell_D,int& n_global) //TODO optimise, this should be done inside meix class
{
  float global_S=0;
  float S;
  n_global=0;
  for (int i=0 ; i<F.size(); i++)
  {
    Point V0(F[i].V0.x,F[i].V0.y,F[i].V0.z);
    Point V1(F[i].V1.x,F[i].V1.y,F[i].V1.z);
    Point V2(F[i].V2.x,F[i].V2.y,F[i].V2.z);

    Point AB=V0-V1;
    Point BC=V1-V2;
    Point CA=V2-V0;

    float a=sqrt(AB.x*AB.x + AB.y*AB.y + AB.z*AB.z);
    float b=sqrt(BC.x*BC.x + BC.y*BC.y + BC.z*BC.z);
    float c=sqrt(CA.x*CA.x + CA.y*CA.y + CA.z*CA.z);

    float s=0.5f*(a+b+c);

    S=sqrt(s*(s-a)*(s-b)*(s-c));

    global_S+=S;

    //we infer already how many cells will fit in that facet
    int dummy=std::round(cell_D*S);
    n_global+=dummy;

//std::cout<<i<<" n= "<<F[i].n<<" cell_density= "<<cell_D<<" S= "<<S<<" nglobal= "<<n_global<<std::endl;

  }
  return global_S;
}


//this function will distribute uniformly random epithelial cells on top of the
//mesh surface.
// template<typename Pt, int n_max, template<typename, int> class Solver>
// void seed_epithelium_on_meix(std::vector<Facet> F, Solution<Pt, n_max, Solver>& bolls, float cell_D, unsigned int n_0 = 0)
// {
//
//     assert(n_0 < *bolls.h_n);
//     int j=0;
// //std::cout<<"crash0.1"<<std::endl;
//     for(int i=0; i<F.size(); i++)
//     {
//       Point V0(F[i].p1x,F[i].p1y,F[i].p1z);
//       Point V1(F[i].p2x,F[i].p2y,F[i].p2z);
//       Point V2(F[i].p3x,F[i].p3y,F[i].p3z);
//       int n=F[i].n;
//       int c=0;
// //std::cout<<"crash0.2"<<" n= "<<n<<"V0x "<<V0.x<<" "<<V0.y<<" "<<V0.z<<std::endl;
//       float phi=atan2(F[i].ny,F[i].nx);
//       float theta= acos(F[i].nz);
// //std::cout<<"crash0.3"<<std::endl;
//       while (c<n)
//       {
//         // 0<s<1 ; 0<t<1 ; s+t<1
//         float s=rand()/(RAND_MAX+1.f);
//         float t=rand()/(RAND_MAX+1.f);
//         if (s+t>1) continue;
//
//         float a=1-s-t;
//
//         //std::cout<<"s= "<<s<<" t= "<<t<<" a= "<<a<<" s+t= "<< s+t <<std::endl;
//
//         Point p= V0*a + V1*s + V2*t;
//
//         bolls.h_X[j].x = p.x;
//         bolls.h_X[j].y = p.y;
//         bolls.h_X[j].z = p.z;
//         bolls.h_X[j].phi = phi;
//         bolls.h_X[j].theta = theta;
// //std::cout<<i<<" crash0.4 j= "<<j<<" c= "<<c<<" n= "<<n<<std::endl;
//         j++; c++;
//       }
//     }
//
// }


//this function will distribute uniformly random epithelial cells on top of the
//mesh surface.
template<typename Pt, int n_max, template<typename, int> class Solver>
void seed_epithelium_on_meix_v2(Meix meix, Solution<Pt, n_max, Solver>& bolls, float cell_D, unsigned int n_0 = 0)
{

    assert(n_0 < *bolls.h_n);

//std::cout<<"crash0.1"<<std::endl;
    int nF=meix.Facets.size();
    //std::cout<<"nF "<<nF<<std::endl;
    for(int i=0; i<*bolls.h_n; i++)
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

        bolls.h_X[i].x = p.x;
        bolls.h_X[i].y = p.y;
        bolls.h_X[i].z = p.z;
        bolls.h_X[i].phi = phi;
        bolls.h_X[i].theta = theta;
//std::cout<<i<<" crash0.4 j= "<<j<<" c= "<<c<<" n= "<<n<<std::endl;
        bingo=true;
      }
    }

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
  // argv[4]=assumed cell radius

  std::string output_tag=argv[1];
  std::string file_name=argv[2];

  //First, load the mesh file so we can get the maximum dimensions of the system
  //std::vector<Facet> facets0; //store original mesh
  //std::vector<Facet> facets;  //store rescaled mesh
  //GetMeixSTL(file_name,facets0);
  Meix meix(file_name);

  //we need to rescale because it's too large
  //std::vector<Facet> facets;
  //float resc=0.05; //sphere
  //float resc=0.02; //limb10.04
  //float resc=0.01; //limb12.04
  //float resc=stof(argv[3]);

  //Compute max length in X axis to know how much we need to rescale
  //**********************************************************************
  //Attention! we are assuming the PD axis of the limb is aligned with X
  //**********************************************************************

  float xmin=10000.0f,xmax=-10000.0f;
  float ymin,ymax,zmin,zmax;
  //for(std::vector<Facet>::iterator fit = facets0.begin(); fit != facets0.end(); fit++)
  for(int i=0 ; i<meix.Facets.size() ; i++)
  {
    //Facet f=*fit;
    //if(f.cx<xmin) xmin=f.cx;  if(f.cx>xmax) xmax=f.cx;
    if(meix.Facets[i].C.x<xmin) xmin=meix.Facets[i].C.x;  if(meix.Facets[i].C.x>xmax) xmax=meix.Facets[i].C.x;
    //if(f.cy<ymin) ymin=f.cy;  if(f.cy>ymax) ymax=f.cy;
    //if(f.cz<zmin) zmin=f.cz;  if(f.cz>zmax) zmax=f.cz;
  }
  float dx=xmax-xmin;
  //float dy=ymax-ymin;
  //float dz=zmax-zmin;

  float target_dx=std::stof(argv[3]);
  float resc=target_dx/dx;
  std::cout<<"xmax= "<<xmax<<" xmin= "<<xmin<<std::endl;
  std::cout<<"dx= "<<dx<<" target_dx= "<<target_dx<<" rescaling factor resc= "<<resc<<std::endl;

  //MeixRescale(facets0, facets, resc);
  meix.Rescale(resc);

  //Compute min. and max, positions in x,y,z from rescaled mesh
  xmin=10000.0f;xmax=-10000.0f;ymin=10000.0f;ymax=-10000.0f;zmin=10000.0f;zmax=-10000.0f;
  //for(std::vector<Facet>::iterator fit = facets.begin(); fit != facets.end(); fit++)
  for(int i=0 ; i<meix.Facets.size() ; i++)
  {
    //Facet f=*fit;
    // if(f.cx<xmin) xmin=f.cx;  if(f.cx>xmax) xmax=f.cx;
    // if(f.cy<ymin) ymin=f.cy;  if(f.cy>ymax) ymax=f.cy;
    // if(f.cz<zmin) zmin=f.cz;  if(f.cz>zmax) zmax=f.cz;
    if(meix.Facets[i].C.x<xmin) xmin=meix.Facets[i].C.x;  if(meix.Facets[i].C.x>xmax) xmax=meix.Facets[i].C.x;
    if(meix.Facets[i].C.y<ymin) ymin=meix.Facets[i].C.y;  if(meix.Facets[i].C.y>ymax) ymax=meix.Facets[i].C.y;
    if(meix.Facets[i].C.z<zmin) zmin=meix.Facets[i].C.z;  if(meix.Facets[i].C.z>zmax) zmax=meix.Facets[i].C.z;
  }
  dx=xmax-xmin;
  float dy=ymax-ymin;
  float dz=zmax-zmin;

  //In order to efficiently seed the mesh surface, we need to estimate epithelial
  //cell density based on the cell size. We have cell radius (bolls eq. radius)
  //as input from cmd. line. We assume hexagonal packing of epithelia, so the
  //effective surface occupied by one cell will be the one of an hexagon with
  //apothem equal to the cell radius

  float cell_radius=std::stof(argv[4]);
  float cell_S=cell_radius*cell_radius*6.f/sqrt(3.f); //regular hexagon formula
  float cell_density=1.f/cell_S;

  //calculate whole Surface area of meix
  //int dummy;
  //float meix_S=meix_get_surfarea(meix.Facets,cell_density,dummy);
  //int n_bolls=dummy;
  int n_bolls=std::round(cell_density*meix.SurfArea);

  std::cout<<"nbolls= "<<n_bolls<<" cell_density= "<<cell_density<<" meix_S= "<<meix.SurfArea<<std::endl;

  Solution<Cell, n_max, Lattice_solver> bolls(n_bolls);

  //seed the cells onto the meix
  seed_epithelium_on_meix_v2(meix, bolls, cell_density);

  Property<n_max, Cell_types> type;
  cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
  for (int i = 0; i < n_bolls; i++)
  {
      type.h_prop[i] = epithelium;
  }
  bolls.copy_to_device();
  type.copy_to_device();

  Property<n_max, int> n_mes_nbs;
  cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
  Property<n_max, int> n_epi_nbs;
  cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));


//std::cout<<"crash2"<<endl;
  //Write fitted morphology on output file
  Vtk_output output(argv[1]);
  //balls.copy_to_host();
  //output.write_positions(bolls);
  //output.write_polarity(bolls);
//std::cout<<"crash3"<<endl;

int relax_time=5000;
int write_interval=1;

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




  //write down the meix in the vtk file to compare it with the posterior seeding
  write_meix_vtk(meix,output_tag);


  //Translocation and rotation of the mesh, if convenient?
  //TODO
  //

  //we use the maximum lengths of the mesh to draw a cube that includes the mesh
  //Let's fill the cube with bolls
  //How many bolls? We calculate the volume of the cube we want to fill
  //then we calculate how many bolls add up to that volume, correcting by the
  //inefficiency of a cubic packing (0.74)----> Well in the end we don't correct cause it wasn't packed enough

  // //const float packing_factor=0.74048f;
  // float cube_vol=dx*dy*dz;
  // float r_boll=0.3f;
  // float boll_vol=4./3.*M_PI*pow(r_boll,3);
  // int n_bolls_cube=cube_vol/boll_vol;
  //
  // std::cout<<"dims "<<dx<<" "<<dy<<" "<<dz<<std::endl;
  // std::cout<<"nbolls in cube "<<n_bolls_cube<<std::endl;
  //
  // Solution<Cell, n_max, Lattice_solver> bolls(n_bolls_cube);
  // //Fill the rectangle with bolls
  // uniform_cubic_rectangle(xmin,ymin,zmin,dx,dy,dz,bolls);
  //
  // Property<n_max, Cell_types> type;
  // cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
  // for (auto i = 0; i < n_bolls_cube; i++)
  // {
  //     type.h_prop[i] = mesenchyme;
  // }
  //
  // bolls.copy_to_device();
  // type.copy_to_device();
  // Property<n_max, int> n_mes_nbs;
  // cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
  // Property<n_max, int> n_epi_nbs;
  // cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));
  //
  // // We run the solver on bolls so the cube of bolls relaxes
  // std::stringstream ass;
  // ass << argv[1] << ".cubic_relaxation";
  // std::string cubic_out = ass.str();
  //
  // int relax_time=stoi(argv[4]);
  // int write_interval=relax_time/10;
  // std::cout<<"relax_time "<<relax_time<<" write interval "<< write_interval<<std::endl;
  //
  // Vtk_output output(cubic_out);
  //
  // for (auto time_step = 0; time_step <= relax_time; time_step++)
  // {
  //   if(time_step%write_interval==0 || time_step==relax_time)
  //   {
  //     bolls.copy_to_host();
  //   }
  //
  //   bolls.build_lattice(r_max);
  //
  //   thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_0, 0);
  //
  //   bolls.take_step<relaxation_force>(dt);
  //
  //   //write the output
  //   if(time_step%write_interval==0 || time_step==relax_time)
  //   {
  //     output.write_positions(bolls);
  //     output.write_polarity(bolls);
  //   }
  //
  // }

  //Find the bolls that are inside the mesh and store their positions
  //METHOD: Shooting a ray from a ball and counting how many triangles intersects.
  //If the ray intersects an even number of facets the boll is out of the mesh, else is in

  // //Setup the list of points
  // std::vector<Point> points;
  // for (auto i = 0; i < n_bolls_cube; i++)
  // {
  //   Point p=Point(bolls.h_X[i].x, bolls.h_X[i].y, bolls.h_X[i].z);
  //   points.push_back(p);
  // }
  //
  // //Setup the list of inclusion test results
  // int* results=new int[n_bolls_cube];
  // //Set direction of ray
  // Point dir=Point(1.0f,0.0f,0.0f);
  //
  // MeixInclusionTest(points , facets , results, dir);
  //
  // //Make a new list with the ones that are inside
  // std::vector<Point> points_fit;
  // int n_bolls_fit=0;
  // for (int i = 0; i < n_bolls_cube; i++)
  // {
  //   if(results[i]==1)
  //   {
  //     points_fit.push_back(points[i]);
  //     n_bolls_fit++;
  //   }
  // }
  //
  // std::cout<<"bolls_in_cube "<<n_bolls_cube<<" bolls after fit "<<n_bolls_fit<<std::endl;
  //
  // Solution<Cell, n_max, Lattice_solver> balls(n_bolls_fit);
  // fill_solver_from_list(points_fit,balls);
  //
  // //Write fitted morphology on output file
  // Vtk_output output_2(argv[1]);
  // balls.copy_to_host();
  // output_2.write_positions(balls);
  // output_2.write_polarity(balls);

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
  //Solution<Cell, n_max, Lattice_solver> meix(facets.size());

  //fill_solver_from_meix(facets,meix);

  //std::stringstream ss;
  //ss << argv[1] << ".centre_meix" ;
  //std::string meix_out = ss.str();

  //Vtk_output output_meix(meix_out);
  //output_meix.write_positions(meix);
  //output_meix.write_polarity(meix);
  //std::cout<<"meix size: number of facets: "<<facets.size()<<std::endl;
  //****************************************************************//

  return 0;
}
