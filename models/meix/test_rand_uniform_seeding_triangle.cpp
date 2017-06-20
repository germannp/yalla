//Test a function to seed uniformly-distributed, random points within a 3D
//triangle. The idea is to define the parametric equations of the triangle,
//then randomise the parameters (s,t) and get P(x,y,z) from those.

//Got the equations from here: http://geomalgorithms.com/a04-_planes.html#Barycentric-Coordinate-Compute

#include <vector>
#include <fstream>
#include <iostream>
#include <assert.h>
#include "../models/meix/meix_inclusion_test.h" //The geometry data structures here will come in handy

//const int n=10;
const float density=1.f;

//Heron's formula
float triangle_area(Point V0, Point V1, Point V2)
{
  Point AB=V0-V1;
  Point BC=V1-V2;
  Point CA=V2-V0;

  float a=sqrt(AB.x*AB.x + AB.y*AB.y + AB.z*AB.z);
  float b=sqrt(BC.x*BC.x + BC.y*BC.y + BC.z*BC.z);
  float c=sqrt(CA.x*CA.x + CA.y*CA.y + CA.z*CA.z);

  float s=0.5f*(a+b+c);

  return sqrt(s*(s-a)*(s-b)*(s-c));
}


int main()
{

  // std::ofstream triangle("output/triangle_vertices.vtk");
  // std::ofstream file("output/triangle_seed.vtk");
  // Point V0 (0.0f, 0.0f, 0.0f);
  // Point V1 (-1.0f, 1.0f, 2.0f);
  // Point V2 (1.0f, 3.0f, -2.0f);

  std::ofstream triangle("output/triangle_vertices_v2.vtk");
  std::ofstream file("output/triangle_seed_v2.vtk");
  Point V0 (0.0f, 0.0f, 0.0f);
  Point V1 (-10.0f, 10.0f, 20.0f);
  Point V2 (10.0f, 30.0f, -20.0f);

  float S=triangle_area(V0,V1,V2);

  int n=density*S;

  std::cout<<"triangle S.A= "<<S<<" density= "<<density<<" n= "<<n<<std::endl;

  std:: vector<Point> points;
  Point p;
  float s,t,a;

  int c=0;
  while (c<n)
  {
    // 0<s<1 ; 0<t<1 ; s+t<1
    s=rand()/(RAND_MAX+1.f);
    t=rand()/(RAND_MAX+1.f);
    if (s+t>1) continue;

    a=1-s-t;

    //std::cout<<"s= "<<s<<" t= "<<t<<" a= "<<a<<" s+t= "<< s+t <<std::endl;

    p= V0*a + V1*s + V2*t;
    points.push_back(p);
    c++;
  }

  //write down on a vtk file


  assert(triangle.is_open());

  triangle << "# vtk DataFile Version 3.0\n";
  triangle << "triangle_vertices" << "\n";
  triangle << "ASCII\n";
  triangle << "DATASET POLYDATA\n";

  triangle << "\nPOINTS " << 3 << " float\n";
  triangle << V0.x << " " << V0.y << " " << V0.z << "\n";
  triangle << V1.x << " " << V1.y << " " << V1.z << "\n";
  triangle << V2.x << " " << V2.y << " " << V2.z << "\n";

  triangle << "\nPOLYGONS " << 1 << " " << 4 << "\n";
  triangle << "3 " << " "<< 0 << " "<< 1 << " "<< 2 << "\n";





  assert(file.is_open());

  file << "# vtk DataFile Version 3.0\n";
  file << "triangle_seed" << "\n";
  file << "ASCII\n";
  file << "DATASET POLYDATA\n";

  file << "\nPOINTS " << n << " float\n";
  for (auto i = 0; i < n; i++)
      file << points[i].x << " " << points[i].y << " " << points[i].z << "\n";

  file << "\nVERTICES " << n << " " << 2*n << "\n";
  for (auto i = 0; i < n; i++)
      file << "1 " << i << "\n";



}
