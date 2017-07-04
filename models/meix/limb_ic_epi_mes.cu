//Makes initial conditions for a limb bud taking the morphology from a 3D model
//(3D mesh), then fills the volume with mesenchymal cells and the surface with
//epithelial cells, then lets teh system relax.

// Command line arguments
// argv[1]=input mesh file name
// argv[2]=output file tag
// argv[3]=target limb bud size (dx)
// argv[4]=cube relax_time
// argv[5]=limb bud relax_time

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

MAKE_PT(Cell, theta, phi);

__device__ Cell relaxation_force(Cell Xi, Cell r, float dist, int i, int j) {
    Cell dF {0};

    if(i==j) return dF;

    if(d_freeze[i]==1) return dF; //frozen cells don't experience force so don't move

    if (dist > r_max) return dF;

    float F;
    if (d_type[i] == d_type[j]) {
        if(d_type[i]==mesenchyme) F = fmaxf(0.8 - dist, 0)*8.f - fmaxf(dist - 0.8, 0)*2.f;
        else F = fmaxf(0.8 - dist, 0)*8.f - fmaxf(dist - 0.8, 0)*8.f;

    } else {
        F = fmaxf(0.9 - dist, 0)*8.f - fmaxf(dist - 0.9, 0)*2.f;
    }
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    if(d_type[i]==epithelium && d_type[j]==epithelium) {
        dF += rigidity_force(Xi, r, dist)*0.15f;
    }

    if (d_type[j] == mesenchyme) atomicAdd(&d_mes_nbs[i],1);

    return dF;
}

__device__ Cell wall_force(Cell Xi, Cell r, float dist, int i, int j) {
    Cell dF {0};

    if(i==j) return dF;

    if (dist > r_max) return dF;

    float k;
    if(r_min-dist>0) //different coefficients for repulsion and adhesion
        k=8.f;
    else
        k=2.f;

    auto F = k*(r_min - dist);
    dF.x = r.x*F/dist;
    dF.y = r.y*F/dist;
    dF.z = r.z*F/dist;

    if(d_type[i]==epithelium && d_type[j]==epithelium) {
        dF += rigidity_force(Xi, r, dist)*0.15f;
    }

    if(Xi.x<0) dF.x=0.f;

    return dF;
}

__device__ float relaxation_friction(Cell Xi, Cell r, float dist, int i, int j) {
    return 0;
}

__device__ float freeze_friction(Cell Xi, Cell r, float dist, int i, int j) {
    if(d_freeze[i]==1) return 0;
    return 1;
}

__device__ float wall_friction(Cell Xi, Cell r, float dist, int i, int j) {
    if(Xi.x<0) return 0;
    return 1;
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

    std::string file_name=argv[1];
    std::string output_tag=argv[2];

    //First, load the mesh file so we can get the maximum dimensions of the system
    Meix meix(file_name);

    //Mesh translation, we're gonna put its centre on the origin of coordinates
    Point old_centroid=meix.Get_centroid();

    meix.Translate(old_centroid*-1.f);

    Point new_centroid=meix.Get_centroid();

    std::cout<<"old centroid= "<<old_centroid.x<<" "<<old_centroid.y<<" "<<old_centroid.z<<std::endl;
    std::cout<<"new centroid= "<<new_centroid.x<<" "<<new_centroid.y<<" "<<new_centroid.z<<std::endl;

    //rotation
    //So the limb boundary with the flank is aligned with the y-z plane
    Point normal0=meix.Facets[0].N; //the first facet is always on the boundary plane for some reason
    float theta0=atan2(normal0.y, normal0.x);
    float phi0=acos(normal0.z);
    float const right_theta=M_PI;
    float const right_phi=M_PI/2.f;
    float correction_theta=right_theta - theta0;
    float correction_phi=right_phi - phi0;

    std::cout<<"theta0, phi0: "<<theta0<<" "<<phi0<<" correction: "<<correction_theta<<" "<<correction_phi<<std::endl;

    meix.Rotate(correction_theta,correction_phi);

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
    meix_mesench.Rescale_absolute(-r_min*1.3);//*1.2

    //Translate the mesh again so the flank boundary coincides with the x=0 plane
    Point vector(meix.Facets[0].C.x*-1.f, 0.f, 0.f);
    meix.Translate(vector);
    meix_mesench.Translate(vector);

    std::cout<<"new x position of flank boundary "<<meix.Facets[0].C.x<<std::endl;

    //Compute min. and max, positions in x,y,z from rescaled mesh
    xmin=10000.0f;xmax=-10000.0f;ymin=10000.0f;ymax=-10000.0f;zmin=10000.0f;zmax=-10000.0f;
    for(int i=0 ; i<meix_mesench.n ; i++) {
      if(meix_mesench.Facets[i].C.x<xmin) xmin=meix_mesench.Facets[i].C.x;  if(meix_mesench.Facets[i].C.x>xmax) xmax=meix_mesench.Facets[i].C.x;
      if(meix_mesench.Facets[i].C.y<ymin) ymin=meix_mesench.Facets[i].C.y;  if(meix_mesench.Facets[i].C.y>ymax) ymax=meix_mesench.Facets[i].C.y;
      if(meix_mesench.Facets[i].C.z<zmin) zmin=meix_mesench.Facets[i].C.z;  if(meix_mesench.Facets[i].C.z>zmax) zmax=meix_mesench.Facets[i].C.z;
    }
    dx=xmax-xmin; dy=ymax-ymin; dz=zmax-zmin;

    //we use the maximum lengths of the mesh to draw a cube that includes the mesh
    //Let's fill the cube with bolls
    //How many bolls? We calculate the volume of the cube we want to fill
    //then we calculate how many bolls add up to that volume, correcting by the
    //inefficiency of a cubic packing (0.74)----> Well in the end we don't correct cause it wasn't packed enough

    //const float packing_factor=0.74048f;
    float cube_vol=dx*dy*dz;
    float r_boll=0.5f*r_min;
    float boll_vol=4./3.*M_PI*pow(r_boll,3);
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

    // We run the solver on bolls so the cube of bolls relaxes
    std::stringstream ass;
    ass << argv[1] << ".cubic_relaxation";
    std::string cubic_out = ass.str();

    int relax_time=std::stoi(argv[4]);
    int skip_step=relax_time/10;
    std::cout<<"relax_time "<<relax_time<<" write interval "<< skip_step<<std::endl;

    Vtk_output cubic_output(cubic_out);

    for (auto time_step = 0; time_step <= relax_time; time_step++) {
        if(time_step%skip_step==0 || time_step==relax_time){
            cube.copy_to_host();
        }

        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_bolls_cube, 0);

        cube.take_step<relaxation_force, relaxation_friction>(dt);

        //write the output
        if(time_step%skip_step==0 || time_step==relax_time) {
            cubic_output.write_positions(cube);
            cubic_output.write_polarity(cube);
        }
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
        if(results2[i]==1){
            type.h_prop[i]=mesenchyme; freeze.h_prop[i]=1;
        }
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
            if(meix.Facets[f].C.x<0.001f){ //the cells contacting the flank boundary can't be epithelial
                type.h_prop[i]=mesenchyme; freeze.h_prop[i]=1;
                continue;
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

    relax_time=std::stoi(argv[5]);
    skip_step=1;//relax_time/10;

    for (auto time_step = 0; time_step <= relax_time; time_step++) {
        if(time_step%skip_step==0 || time_step==relax_time) {
            bolls.copy_to_host();
        }

        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_bolls_total, 0);

        bolls.take_step<relaxation_force, freeze_friction>(dt);

        //write the output
        if(time_step%skip_step==0 || time_step==relax_time) {
            output.write_positions(bolls);
            output.write_polarity(bolls);
            output.write_property(type);
            output.write_property(freeze);
        }

    }

    //Unfreeze the mesenchyme
    for(int i=0 ; i<n_bolls_total ; i++) {
        if(type.h_prop[i]==mesenchyme)
            freeze.h_prop[i]=0;
    }

    freeze.copy_to_device();

    Vtk_output output_unfrozen(output_tag+".unfrozen");
    skip_step=1;
    //try relaxation with unfrozen mesenchyme
    for (auto time_step = 0; time_step <= relax_time; time_step++) {
        if(time_step%skip_step==0 || time_step==relax_time)
            bolls.copy_to_host();

        thrust::fill(thrust::device, n_mes_nbs.d_prop, n_mes_nbs.d_prop + n_bolls_total, 0);

        bolls.take_step<wall_force, wall_friction>(dt);

        //write the output
        if(time_step%skip_step==0 || time_step==relax_time) {
            output_unfrozen.write_positions(bolls);
            output_unfrozen.write_polarity(bolls);
            output_unfrozen.write_property(type);
            output_unfrozen.write_property(freeze);
        }

    }

    //write down the meix in the vtk file to compare it with the posterior seeding
    meix.WriteVtk(output_tag);
    //write down the mesenchymal mesh in the vtk file to compare it with the posterior filling
    meix_mesench.WriteVtk(output_tag+".mesench");

    //Create a dummy meix that depicts the x=0 plane, depicting the flank boundary
    Meix wall;
    Point A(0.f,2*ymin,2*zmin);
    Point B(0.f,2*ymin,2*zmax);
    Point C(0.f,2*ymax,2*zmin);
    Point D(0.f,2*ymax,2*zmax);
    Point N(1.f,0.f,0.f);
    Triangle ABC(A,B,C,N);
    Triangle BCD(B,C,D,N);
    wall.n=2;
    wall.Facets.push_back(ABC);
    wall.Facets.push_back(BCD);
    wall.WriteVtk(output_tag+".wall");

    return 0;
}
