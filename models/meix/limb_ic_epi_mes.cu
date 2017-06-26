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
const auto r_min=0.8;
const auto dt = 0.05*r_min*r_min;
const auto n_max = 150000;

enum Cell_types {mesenchyme, epithelium};

__device__ Cell_types* d_type;
__device__ int* d_freeze;
__device__ int* d_mes_nbs;
__device__ int* d_epi_nbs;

MAKE_PT(Cell, theta, phi);

__device__ Cell relaxation_force(Cell Xi, Cell r, float dist, int i, int j) {
    Cell dF {0};

    if(i==j) return dF;

    if(d_freeze[i]==1) return dF; //frozen cells don't experience force so don't move

    if (dist > r_max) return dF;

    float k;
    if(r_min-dist>0) //different coefficients for repulsion and adhesion
    {
      if(d_type[i]==epithelium && d_type[j]==mesenchyme)
      {k=8.f;} //epi. mesench. contact
      else
      {k=8.f;} //any other contact
    }else{
      if(d_type[i]==epithelium && d_type[j]==mesenchyme)
      {k=2.f;} //epi. mesench. contact
      else
      {k=2.f;} //any other contact
    }
    auto F = k*(r_min - dist);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    if(d_type[i]==epithelium && d_type[j]==epithelium)
    {
        dF += rigidity_force(Xi, r, dist)*0.3f;//*3;
    }

    if (d_type[j] == epithelium) {atomicAdd(&d_epi_nbs[i],1);}
    else {atomicAdd(&d_mes_nbs[i],1);}

    return dF;
}

__device__ float freeze_friction(Cell Xi, Cell r, float dist, int i, int j) {
    if(d_freeze[i]==1) return 1;
    return 0;
}

// Distribute bolls uniformly random in rectangular cube
template<typename Pt, int n_max, template<typename, int> class Solver>
void uniform_cubic_rectangle(float x0,float y0,float z0,float dx,float dy,float dz, Solution<Pt, n_max, Solver>& bolls, unsigned int n_0 = 0) {
    assert(n_0 < *bolls.h_n);

    for (auto i = n_0; i < *bolls.h_n; i++) {
        bolls.h_X[i].x = x0+dx*(rand()/(RAND_MAX+1.));
        bolls.h_X[i].y = y0+dy*(rand()/(RAND_MAX+1.));
        bolls.h_X[i].z = z0+dz*(rand()/(RAND_MAX+1.));
        bolls.h_X[i].phi=0.0f;
        bolls.h_X[i].theta=0.0f;
    }

    bolls.copy_to_device();
}

//*****************************************************************************

int main(int argc, char const *argv[]) {
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

    //Mesh translation, we're gonna put its centre on the origin of coordinates
    //first, calculate centroid of the mesh
    Point centroid;
    for (int i=0 ; i<meix.n ; i++) {
        centroid=centroid+meix.Facets[i].C;
    }
    centroid=centroid*(1.f/float(meix.n));

    //translation
    Point new_centroid;
    for (int i=0 ; i<meix.n ; i++) {
        meix.Facets[i].V0=meix.Facets[i].V0-centroid;
        meix.Facets[i].V1=meix.Facets[i].V1-centroid;
        meix.Facets[i].V2=meix.Facets[i].V2-centroid;
        meix.Facets[i].C=meix.Facets[i].C-centroid;
        new_centroid=new_centroid+meix.Facets[i].C;
    }

    new_centroid=new_centroid*(1.f/float(meix.n));

    std::cout<<"old centroid= "<<centroid.x<<" "<<centroid.y<<" "<<centroid.z<<std::endl;
    std::cout<<"new centroid= "<<new_centroid.x<<" "<<new_centroid.y<<" "<<new_centroid.z<<std::endl;

    //rotation
    //So the limb boundary with the flank is aligned with the y-z plane
    Point normal0=meix.Facets[0].N; //the first facet is always on the boundary plane for some reason
    float theta0=atan2(normal0.y, normal0.x);
    float phi0=acos(normal0.z);
    float const right_theta=M_PI;
    float const right_phi=M_PI/2.f;
    float correction_theta=right_theta-theta0;
    float correction_phi=right_phi-phi0;

    std::cout<<"theta0, phi0: "<<theta0<<" "<<phi0<<" correction: "<<correction_theta<<" "<<correction_phi<<std::endl;

    //rotation around z axis (theta correction)
    std::cout<<"normal of facet 0 before theta rotation: "<<meix.Facets[0].N.x<<" "<<meix.Facets[0].N.y<<" "<<meix.Facets[0].N.z<<std::endl;
    for(int i=0 ; i<meix.n ; i++) {
        Point old=meix.Facets[i].V0;
        meix.Facets[i].V0.x=old.x*cos(correction_theta)-old.y*sin(correction_theta);
        meix.Facets[i].V0.y=old.x*sin(correction_theta)+old.y*cos(correction_theta);

        old=meix.Facets[i].V1;
        meix.Facets[i].V1.x=old.x*cos(correction_theta)-old.y*sin(correction_theta);
        meix.Facets[i].V1.y=old.x*sin(correction_theta)+old.y*cos(correction_theta);

        old=meix.Facets[i].V2;
        meix.Facets[i].V2.x=old.x*cos(correction_theta)-old.y*sin(correction_theta);
        meix.Facets[i].V2.y=old.x*sin(correction_theta)+old.y*cos(correction_theta);

        old=meix.Facets[i].C;
        meix.Facets[i].C.x=old.x*cos(correction_theta)-old.y*sin(correction_theta);
        meix.Facets[i].C.y=old.x*sin(correction_theta)+old.y*cos(correction_theta);
        //Recalculate normal
        Point v=meix.Facets[i].V1-meix.Facets[i].V0;
        Point u=meix.Facets[i].V2-meix.Facets[i].V0;
        Point n(u.y*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.y*v.x);
        float d=sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        meix.Facets[i].N=n*(1.f/d);
    }
    std::cout<<"normal of facet 0 after theta rotation: "<<meix.Facets[0].N.x<<" "<<meix.Facets[0].N.y<<" "<<meix.Facets[0].N.z<<std::endl;

    //rotation around x axis (phi correction)
    for(int i=0 ; i<meix.n ; i++) {
        Point old=meix.Facets[i].V0;
        meix.Facets[i].V0.x=old.x*cos(correction_phi)-old.z*sin(correction_phi);
        meix.Facets[i].V0.z=old.x*sin(correction_phi)+old.z*cos(correction_phi);

        old=meix.Facets[i].V1;
        meix.Facets[i].V1.x=old.x*cos(correction_phi)-old.z*sin(correction_phi);
        meix.Facets[i].V1.z=old.x*sin(correction_phi)+old.z*cos(correction_phi);

        old=meix.Facets[i].V2;
        meix.Facets[i].V2.x=old.x*cos(correction_phi)-old.z*sin(correction_phi);
        meix.Facets[i].V2.z=old.x*sin(correction_phi)+old.z*cos(correction_phi);

        old=meix.Facets[i].C;
        meix.Facets[i].C.x=old.x*cos(correction_phi)-old.z*sin(correction_phi);
        meix.Facets[i].C.z=old.x*sin(correction_phi)+old.z*cos(correction_phi);
        //Recalculate normal
        Point v=meix.Facets[i].V1-meix.Facets[i].V0;
        Point u=meix.Facets[i].V2-meix.Facets[i].V0;
        Point n(u.y*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.y*v.x);
        float d=sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        meix.Facets[i].N=n*(1.f/d);
    }
    std::cout<<"normal of facet 0 after phi rotation: "<<meix.Facets[0].N.x<<" "<<meix.Facets[0].N.y<<" "<<meix.Facets[0].N.z<<std::endl;

    //Compute max length in X axis to know how much we need to rescale
    //**********************************************************************
    //Attention! we are assuming the PD axis of the limb is aligned with X
    //**********************************************************************

    float xmin=10000.0f,xmax=-10000.0f;
    float ymin,ymax,zmin,zmax;
    float dx,dy,dz;

    for(int i=0 ; i<meix.n ; i++) {
        if(meix.Facets[i].C.x<xmin) xmin=meix.Facets[i].C.x;  if(meix.Facets[i].C.x>xmax) xmax=meix.Facets[i].C.x;
    }
    dx=xmax-xmin;

    float target_dx=std::stof(argv[3]);
    float resc=target_dx/dx;
    std::cout<<"xmax= "<<xmax<<" xmin= "<<xmin<<std::endl;
    std::cout<<"dx= "<<dx<<" target_dx= "<<target_dx<<" rescaling factor resc= "<<resc<<std::endl;

    //meix defines the overall shape of the limb bud
    meix.Rescale_relative(resc);
    //meix_mesench defines the volume occupied by the mesenchyme (smaller than meix)
    //meix_mesench will be transformed from the main meix (rescaled)
    Meix meix_mesench=meix;
    meix_mesench.Rescale_absolute(-r_min*1.2);

    //Compute min. and max, positions in x,y,z from rescaled mesh
    xmin=10000.0f;xmax=-10000.0f;ymin=10000.0f;ymax=-10000.0f;zmin=10000.0f;zmax=-10000.0f;
    for(int i=0 ; i<meix_mesench.n ; i++) {
      if(meix_mesench.Facets[i].C.x<xmin) xmin=meix_mesench.Facets[i].C.x;  if(meix_mesench.Facets[i].C.x>xmax) xmax=meix_mesench.Facets[i].C.x;
      if(meix_mesench.Facets[i].C.y<ymin) ymin=meix_mesench.Facets[i].C.y;  if(meix_mesench.Facets[i].C.y>ymax) ymax=meix_mesench.Facets[i].C.y;
      if(meix_mesench.Facets[i].C.z<zmin) zmin=meix_mesench.Facets[i].C.z;  if(meix_mesench.Facets[i].C.z>zmax) zmax=meix_mesench.Facets[i].C.z;
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
    float r_boll=0.5f*r_min;
    float boll_vol=4./3.*M_PI*pow(r_boll,3);//packing_factor;
    int n_bolls_cube=cube_vol/boll_vol;

    std::cout<<"cube dims "<<dx<<" "<<dy<<" "<<dz<<std::endl;
    std::cout<<"nbolls in cube "<<n_bolls_cube<<std::endl;

    Solution<Cell, n_max, Grid_solver> cube(n_bolls_cube);
    //Fill the rectangle with bolls
    uniform_cubic_rectangle(xmin,ymin,zmin,dx,dy,dz,cube);

    //Variable indicating cell type
    Property<n_max, Cell_types> type;
    cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
    //Variable that indicates which cells are 'frozen', so don't move
    Property<n_max, int> freeze("freeze");
    cudaMemcpyToSymbol(d_freeze, &freeze.d_prop, sizeof(d_freeze));

    for (auto i = 0; i < n_bolls_cube; i++) {
        type.h_prop[i] = mesenchyme;
        freeze.h_prop[i] = 0;
    }

    cube.copy_to_device();
    type.copy_to_device();
    freeze.copy_to_device();

    Property<n_max, int> n_mes_nbs;
    cudaMemcpyToSymbol(d_mes_nbs, &n_mes_nbs.d_prop, sizeof(d_mes_nbs));
    Property<n_max, int> n_epi_nbs;
    cudaMemcpyToSymbol(d_epi_nbs, &n_epi_nbs.d_prop, sizeof(d_epi_nbs));

    // We run the solver on bolls so the cube of bolls relaxes
    std::stringstream ass;
    ass << argv[1] << ".cubic_relaxation";
    std::string cubic_out = ass.str();

    int relax_time=std::stoi(argv[4]);
    int write_interval=relax_time/10;
    std::cout<<"relax_time "<<relax_time<<" write interval "<< write_interval<<std::endl;

    Vtk_output cubic_output(cubic_out);

    for (auto time_step = 0; time_step <= relax_time; time_step++) {
        if(time_step%write_interval==0 || time_step==relax_time)
        {
            cube.copy_to_host();
        }

        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + n_bolls_cube, 0);
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_bolls_cube, 0);

<<<<<<< HEAD
        cube.take_step<relaxation_force, freeze_friction>(dt);

        //write the output
        if(time_step%write_interval==0 || time_step==relax_time) {
            cubic_output.write_positions(cube);
            cubic_output.write_polarity(cube);
        }
=======
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

  //Mesh translation, we're gonna put its centre on the origin of coordinates
  //first, calculate centroid of the mesh
  Point centroid;
  for (int i=0 ; i<meix.n ; i++)
  {
    centroid=centroid+meix.Facets[i].C;
  }
  centroid=centroid*(1.f/float(meix.n));
  //translation
  Point new_centroid;
  for (int i=0 ; i<meix.n ; i++)
  {
    meix.Facets[i].V0=meix.Facets[i].V0-centroid;
    meix.Facets[i].V1=meix.Facets[i].V1-centroid;
    meix.Facets[i].V2=meix.Facets[i].V2-centroid;
    meix.Facets[i].C=meix.Facets[i].C-centroid;
    new_centroid=new_centroid+meix.Facets[i].C;
  }

  new_centroid=new_centroid*(1.f/float(meix.n));

  std::cout<<"old centroid= "<<centroid.x<<" "<<centroid.y<<" "<<centroid.z<<std::endl;
  std::cout<<"new centroid= "<<new_centroid.x<<" "<<new_centroid.y<<" "<<new_centroid.z<<std::endl;

  //Compute max length in X axis to know how much we need to rescale
  //**********************************************************************
  //Attention! we are assuming the PD axis of the limb is aligned with X
  //**********************************************************************

  float xmin=10000.0f,xmax=-10000.0f;
  float ymin,ymax,zmin,zmax;
  float dx,dy,dz;

  for(int i=0 ; i<meix.n ; i++)
  {
    if(meix.Facets[i].C.x<xmin) xmin=meix.Facets[i].C.x;  if(meix.Facets[i].C.x>xmax) xmax=meix.Facets[i].C.x;
  }
  dx=xmax-xmin;

  float target_dx=std::stof(argv[3]);
  float resc=target_dx/dx;
  std::cout<<"xmax= "<<xmax<<" xmin= "<<xmin<<std::endl;
  std::cout<<"dx= "<<dx<<" target_dx= "<<target_dx<<" rescaling factor resc= "<<resc<<std::endl;

  //meix is the reference shape, and will be used to seed the epithelium
  meix.Rescale_relative(resc);
  //meix_mesench will be transformed from the main meix (rescaled), and will be
  //used to fill with mesenchyme
  Meix meix_mesench=meix;
  //meix_mesench.Rescale_absolute(-0.4f);
  meix_mesench.Rescale_relative(0.9f);

  //Compute min. and max, positions in x,y,z from rescaled mesh
  xmin=10000.0f;xmax=-10000.0f;ymin=10000.0f;ymax=-10000.0f;zmin=10000.0f;zmax=-10000.0f;
  for(int i=0 ; i<meix_mesench.n ; i++)
  {
    if(meix_mesench.Facets[i].C.x<xmin) xmin=meix_mesench.Facets[i].C.x;  if(meix_mesench.Facets[i].C.x>xmax) xmax=meix_mesench.Facets[i].C.x;
    if(meix_mesench.Facets[i].C.y<ymin) ymin=meix_mesench.Facets[i].C.y;  if(meix_mesench.Facets[i].C.y>ymax) ymax=meix_mesench.Facets[i].C.y;
    if(meix_mesench.Facets[i].C.z<zmin) zmin=meix_mesench.Facets[i].C.z;  if(meix_mesench.Facets[i].C.z>zmax) zmax=meix_mesench.Facets[i].C.z;
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

  Solution<Cell, n_max, Grid_solver> cube(n_bolls_cube);
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

  for (auto time_step = 0; time_step <= relax_time; time_step++)
  {
    if(time_step%write_interval==0 || time_step==relax_time)
    {
      cube.copy_to_host();
    }

    cube.build_grid(r_max);
>>>>>>> 1a6c21a688a9438c74d0f242f5cc2190b298ab64

    }

    //Find the bolls that are inside the mesh and store their positions
    //METHOD: Shooting a ray from a ball and counting how many triangles intersects.
    //If the ray intersects an even number of facets the boll is out of the mesh, else is in

    //Setup the list of points
    std::vector<Point> points;
    for (auto i = 0; i < n_bolls_cube; i++) {
        Point p=Point(cube.h_X[i].x, cube.h_X[i].y, cube.h_X[i].z);
        points.push_back(p);
    }

    //Setup the list of inclusion test results
    int* results=new int[n_bolls_cube];
    //Set direction of ray
    Point dir=Point(0.0f,1.0f,0.0f);

    meix.InclusionTest(points , results, dir);

    //Make a new list with the ones that are inside
    std::vector<Point> mes_cells;
    int n_bolls_mes=0;
    for (int i = 0; i < n_bolls_cube; i++) {
        if(results[i]==1) {
            mes_cells.push_back(points[i]);
            n_bolls_mes++;
        }
    }

    std::cout<<"bolls_in_cube "<<n_bolls_cube<<" bolls after fill "<<n_bolls_mes<<std::endl;

    //We have filled the whole meix with mesenchyme, now we will convert the most
    //superficial cells to epithelium. We make another inclusion test, this time
    //with the smaller meix_mesench. The cells that fall outside this one (hopefully
    //one cell layer will be converted to epithelial type.

    //Setup the list of inclusion test results
    int* results2=new int[n_bolls_mes];

    meix_mesench.InclusionTest(mes_cells , results2, dir);

    int n_bolls_total=n_bolls_mes;
    Solution<Cell, n_max, Grid_solver> bolls(n_bolls_total);

    //Make a new list with the ones that are inside
    for (int i = 0; i < n_bolls_total; i++) {
        bolls.h_X[i].x=mes_cells[i].x ; bolls.h_X[i].y=mes_cells[i].y ; bolls.h_X[i].z=mes_cells[i].z ;
        if(results2[i]==1){type.h_prop[i]=mesenchyme; freeze.h_prop[i]=1;}
        else{
            type.h_prop[i]=epithelium;
            freeze.h_prop[i]=0;
            //polarity
            Point p=mes_cells[i];
            int f=-1;
            float dmin=1000000.f;
            //we use the closest facet on meix to determine the polarity of the
            //epithelial cell
            for(int j=0 ; j<meix.n ; j++){
                Point r=p-meix.Facets[j].C;
                float d=sqrt(r.x*r.x+r.y*r.y+r.z*r.z);
                if(d<dmin){dmin=d; f=j;}
            }
            bolls.h_X[i].phi = atan2(meix.Facets[f].N.y,meix.Facets[f].N.x);
            bolls.h_X[i].theta = acos(meix.Facets[f].N.z);
        }
    }

    bolls.copy_to_device();
    type.copy_to_device();
    freeze.copy_to_device();

    std::cout<<"n_bolls_total= "<<n_bolls_total<<std::endl;

    Vtk_output output(output_tag);

    relax_time=std::stoi(argv[6]);
    write_interval=1;//relax_time/10;

    for (auto time_step = 0; time_step <= relax_time; time_step++) {
        if(time_step%write_interval==0 || time_step==relax_time) {
            bolls.copy_to_host();
        }

        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + n_bolls_total, 0);
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_bolls_total, 0);

        bolls.take_step<relaxation_force, freeze_friction>(dt);

        //write the output
        if(time_step%write_interval==0 || time_step==relax_time) {
            output.write_positions(bolls);
            output.write_polarity(bolls);
            output.write_property(type);
            output.write_property(freeze);
        }

<<<<<<< HEAD
    }
=======
  int n_bolls_total=n_bolls_mes+n_bolls_epi;
  Solution<Cell, n_max, Grid_solver> bolls(n_bolls_total);
  // Property<n_max, Cell_types> type;
  // cudaMemcpyToSymbol(d_type, &type.d_prop, sizeof(d_type));
>>>>>>> 1a6c21a688a9438c74d0f242f5cc2190b298ab64

    //De-freeze the mesenchyme
    for(int i=0 ; i<n_bolls_total ; i++) {
        if(type.h_prop[i]==mesenchyme)
            freeze.h_prop[i]=0;
    }

    freeze.copy_to_device();

    //We eliminate the epithelial cells that have separated from the mesenchyme
    bolls.copy_to_host();
    n_mes_nbs.copy_to_host();
    std::vector<Cell> epi_trimmed;
    std::vector<Cell_types> new_type;
    int n_bolls_trimmed=0;
    for(int i=0 ; i<n_bolls_total ; i++) {
        if(type.h_prop[i]==epithelium && n_mes_nbs.h_prop[i]==0)
            continue;

        new_type.push_back(type.h_prop[i]);
        epi_trimmed.push_back(bolls.h_X[i]);
        n_bolls_trimmed++;
    }

    Solution<Cell, n_max, Grid_solver> bolls_trimmed(n_bolls_trimmed);
    for(int i=0 ; i<n_bolls_trimmed ; i++) {
        bolls_trimmed.h_X[i]=epi_trimmed[i];
        type.h_prop[i]=new_type[i];
    }

    std::cout<<"we got "<<n_bolls_trimmed<<" epi. cells"<<std::endl;

    bolls_trimmed.copy_to_device();
    type.copy_to_device();

    Vtk_output output_trimmed(output_tag+".trimmed");

<<<<<<< HEAD
    //try relaxation without mesenchyme frozen
    for (auto time_step = 0; time_step <= relax_time; time_step++) {
        if(time_step%write_interval==0 || time_step==relax_time)
            bolls_trimmed.copy_to_host();
=======
  bolls.build_grid(r_max);
>>>>>>> 1a6c21a688a9438c74d0f242f5cc2190b298ab64

        thrust::fill(thrust::device, n_epi_nbs.d_prop, n_epi_nbs.d_prop + n_bolls_total, 0);
        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_bolls_total, 0);

        bolls_trimmed.take_step<relaxation_force, freeze_friction>(dt);

        //write the output
        if(time_step%write_interval==0 || time_step==relax_time) {
            output_trimmed.write_positions(bolls_trimmed);
            output_trimmed.write_polarity(bolls_trimmed);
            output_trimmed.write_property(type);
            output_trimmed.write_property(freeze);
        }

    }

    //write down the meix in the vtk file to compare it with the posterior seeding
    meix.WriteVtk(output_tag);
    //write down the mesenchymal mesh in the vtk file to compare it with the posterior filling
    meix_mesench.WriteVtk(output_tag+".mesench");

    return 0;
}
