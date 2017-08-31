//Makes initial conditions for a limb bud taking the morphology from a 3D model
//(3D mesh), then fills the volume with mesenchymal cells and the surface with
//epithelial cells, then lets teh system relax.

// Command line arguments
// argv[1]=input mesh file name
// argv[2]=output file tag
// argv[3]=target limb bud size (dx)


#include "../../include/dtypes.cuh"
#include "../../include/inits.cuh"
#include "../../include/solvers.cuh"
#include "../../include/vtk.cuh"
#include "../../include/meix.cuh"
#include "../../include/polarity.cuh"
#include "../../include/property.cuh"
#include <sstream>
#include <string>
#include <list>
#include <vector>
#include <iostream>


const auto n_max = 150000;

MAKE_PT(Cell, theta, phi);

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
    for(int i=0 ; i<meix.n ; i++) {
        if(meix.Facets[i].C.x<xmin) xmin=meix.Facets[i].C.x;  if(meix.Facets[i].C.x>xmax) xmax=meix.Facets[i].C.x;
    }
    float dx=xmax-xmin;

    float target_dx=std::stof(argv[3]);
    float resc=target_dx/dx;
    std::cout<<"xmax= "<<xmax<<" xmin= "<<xmin<<std::endl;
    std::cout<<"dx= "<<dx<<" target_dx= "<<target_dx<<" rescaling factor resc= "<<resc<<std::endl;

    //meix defines the overall shape of the limb bud
    meix.Rescale_relative(resc);
    //Translate the mesh again so the flank boundary coincides with the x=0 plane
    Point vector(meix.Facets[0].C.x*-1.f, 0.f, 0.f);
    meix.Translate(vector);

    std::cout<<"new x position of flank boundary "<<meix.Facets[0].C.x<<std::endl;

    int i=0;
    while(i<meix.Facets.size()){
        if(meix.Facets[i].N.x>-1.01f && meix.Facets[i].N.x<-0.99f)
            meix.Facets.erase(meix.Facets.begin()+i);
        else
            i++;
    }
    meix.n=meix.Facets.size();

    Solution<Cell, n_max, Grid_solver> shape(meix.n);

    for (int i=0 ; i<meix.n ; i++) {
        Triangle f=meix.Facets[i];
        float r=sqrt(pow(f.N.x,2)+pow(f.N.y,2)+pow(f.N.z,2));
        shape.h_X[i].x = f.C.x;
        shape.h_X[i].y = f.C.y;
        shape.h_X[i].z = f.C.z;
        shape.h_X[i].phi = atan2(f.N.y,f.N.x);
        shape.h_X[i].theta = acos(f.N.z/r);
    }
    Vtk_output meix_output(output_tag);
    meix_output.write_positions(shape);
    meix_output.write_polarity(shape);

    return 0;
}
