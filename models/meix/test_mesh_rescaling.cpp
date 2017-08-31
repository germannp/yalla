#include <sstream>
#include <string>
#include <list>
#include <vector>
#include <iostream>

#include "../../include/meix.cuh"

int main(int argc, char const *argv[])
{
  //we test different types of mesh rescaling (absolute and relative)
  std::string mesh_filename=argv[1];

  Meix meix0(mesh_filename);

  //Mesh translation, we're gonna put its centre on the origin of coordinates
  //first, calculate centroid of the mesh
  Point centroid;
  for (int i=0 ; i<meix0.n ; i++)
  {
    centroid=centroid+meix0.Facets[i].C;
  }
  centroid=centroid*(1.f/float(meix0.n));
  //translation
  Point new_centroid;
  for (int i=0 ; i<meix0.n ; i++)
  {
    meix0.Facets[i].V0=meix0.Facets[i].V0-centroid;
    meix0.Facets[i].V1=meix0.Facets[i].V1-centroid;
    meix0.Facets[i].V2=meix0.Facets[i].V2-centroid;
    meix0.Facets[i].C=meix0.Facets[i].C-centroid;
    new_centroid=new_centroid+meix0.Facets[i].C;
  }

  new_centroid=new_centroid*(1.f/float(meix0.n));

  std::cout<<"old centroid= "<<centroid.x<<" "<<centroid.y<<" "<<centroid.z<<std::endl;
  std::cout<<"new centroid= "<<new_centroid.x<<" "<<new_centroid.y<<" "<<new_centroid.z<<std::endl;

  Meix meix_rel=meix0;
  Meix meix_abs=meix0;

  //check dimensions
  //Compute min. and max, positions in x,y,z from rescaled mesh
  float xmin=10000.0f, xmax=-10000.0f, ymin=10000.0f, ymax=-10000.0f, zmin=10000.0f, zmax=-10000.0f;
  for(int i=0 ; i<meix0.n ; i++)
  {
    if(meix0.Facets[i].C.x<xmin) xmin=meix0.Facets[i].C.x;  if(meix0.Facets[i].C.x>xmax) xmax=meix0.Facets[i].C.x;
    if(meix0.Facets[i].C.y<ymin) ymin=meix0.Facets[i].C.y;  if(meix0.Facets[i].C.y>ymax) ymax=meix0.Facets[i].C.y;
    if(meix0.Facets[i].C.z<zmin) zmin=meix0.Facets[i].C.z;  if(meix0.Facets[i].C.z>zmax) zmax=meix0.Facets[i].C.z;
  }
  float dx=xmax-xmin;
  float dy=ymax-ymin;
  float dz=zmax-zmin;
  std::cout<<"meix 0 dx= "<<dx<<" dy= "<<dy<<" dz= "<<dz<<std::endl;


  meix0.WriteVtk("rescale0");

  meix_rel.Rescale_relative(0.9f);
  meix_rel.WriteVtk("rescale_rel");

  //check dimensions
  //Compute min. and max, positions in x,y,z from rescaled mesh
  xmin=10000.0f; xmax=-10000.0f; ymin=10000.0f; ymax=-10000.0f; zmin=10000.0f; zmax=-10000.0f;
  for(int i=0 ; i<meix_rel.n ; i++)
  {
    if(meix_rel.Facets[i].C.x<xmin) xmin=meix_rel.Facets[i].C.x;  if(meix_rel.Facets[i].C.x>xmax) xmax=meix_rel.Facets[i].C.x;
    if(meix_rel.Facets[i].C.y<ymin) ymin=meix_rel.Facets[i].C.y;  if(meix_rel.Facets[i].C.y>ymax) ymax=meix_rel.Facets[i].C.y;
    if(meix_rel.Facets[i].C.z<zmin) zmin=meix_rel.Facets[i].C.z;  if(meix_rel.Facets[i].C.z>zmax) zmax=meix_rel.Facets[i].C.z;
  }
  dx=xmax-xmin;
  dy=ymax-ymin;
  dz=zmax-zmin;
  std::cout<<"meix_rel dx= "<<dx<<" dy= "<<dy<<" dz= "<<dz<<std::endl;

  meix_abs.Rescale_absolute(-200.f);
  meix_abs.WriteVtk("rescale_abs");

  //check dimensions
  //Compute min. and max, positions in x,y,z from rescaled mesh
  xmin=10000.0f; xmax=-10000.0f; ymin=10000.0f; ymax=-10000.0f; zmin=10000.0f; zmax=-10000.0f;
  for(int i=0 ; i<meix_abs.n ; i++)
  {
    if(meix_abs.Facets[i].C.x<xmin) xmin=meix_abs.Facets[i].C.x;  if(meix_abs.Facets[i].C.x>xmax) xmax=meix_abs.Facets[i].C.x;
    if(meix_abs.Facets[i].C.y<ymin) ymin=meix_abs.Facets[i].C.y;  if(meix_abs.Facets[i].C.y>ymax) ymax=meix_abs.Facets[i].C.y;
    if(meix_abs.Facets[i].C.z<zmin) zmin=meix_abs.Facets[i].C.z;  if(meix_abs.Facets[i].C.z>zmax) zmax=meix_abs.Facets[i].C.z;
  }
  dx=xmax-xmin;
  dy=ymax-ymin;
  dz=zmax-zmin;
  std::cout<<"meix_abs dx= "<<dx<<" dy= "<<dy<<" dz= "<<dz<<std::endl;


  return 0;

}
