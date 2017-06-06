#include <fstream>
#include <list>
#include <vector>
#include <string>
#include "meix.h"
#include "strtk.hpp"   // http://www.partow.net/programming/strtk

using namespace std;

const char *whitespace    = " \t\r\n\f";

//Overloaded operators for Point
   bool operator == (Point& a, Point& b)
  {
    //bool test;
    if(a.x == b.x && a.y == b.y && a.z == b.z)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  bool operator != (Point& a, Point& b)
  {
    //bool test;
    if(a.x != b.x && a.y != b.y && a.z != b.z)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  Point operator + (Point a, Point b)
  {
    Point p;
    p.x=a.x + b.x;
    p.y=a.y + b.y;
    p.z=a.z + b.z;
    return p;
  }

  Point operator - (Point a, Point b)
  {
    Point p;
    p.x=a.x - b.x;
    p.y=a.y - b.y;
    p.z=a.z - b.z;
    return p;
  }

  Point operator * (Point a, float s)
  {
    Point p;
    p.x=a.x * s;
    p.y=a.y * s;
    p.z=a.z * s;
    return p;
  }

//******************************************************************

#define SMALL_NUM   0.00000001 // anything that avoids division overflow
// dot product (3D) which allows vector operations in arguments
#define dot(u,v)   ((u).x * (v).x + (u).y * (v).y + (u).z * (v).z)

int intersect3D_RayTriangle( Ray R, Triangle T, Point* I )
{

    Point    u, v, n;              // triangle vectors
    Point    dir, w0, w;           // ray vectors
    float    r, a, b;              // params to calc ray-plane intersect

    // Point p_v0=T.V0;
    // Point p_v1=T.V1;
    // Point p_v2=T.V2;

    //Triangle T(p_v0, p_v1, p_v2);

    // get triangle edge vectors and plane normal
    u = T.V1 - T.V0;
    v = T.V2 - T.V0;
    //n = u * v;              // cross product
    n=T.N;
    //n.x=F.nx ; n.y=F.ny ; n.z=F.nz;
    //if (n == (Vector)0)             // triangle is degenerate
    if (n.x == 0.0f && n.y==0.0f && n.z==0.0f)             // triangle is degenerate
        return -1;                  // do not deal with this case

    dir = R.P1 - R.P0;              // ray direction vector
    w0 = R.P0 - T.V0;
    a = -dot(n,w0);
    b = dot(n,dir);
    if (fabs(b) < SMALL_NUM) {     // ray is  parallel to triangle plane
        if (a == 0)                 // ray lies in triangle plane
            return 2;
        else return 0;              // ray disjoint from plane
    }

    // get intersect point of ray with triangle plane
    r = a / b;
    if (r < 0.0)                    // ray goes away from triangle
        return 0;                   // => no intersect
    // for a segment, also test if (r > 1.0) => no intersect

    *I = R.P0 + (dir * r);            // intersect point of ray and plane

    // is I inside T?
    float    uu, uv, vv, wu, wv, D;
    uu = dot(u,u);
    uv = dot(u,v);
    vv = dot(v,v);
    w = *I - T.V0;
    wu = dot(w,u);
    wv = dot(w,v);
    D = uv * uv - uu * vv;

    // get and test parametric coords
    float s, t;
    s = (uv * wv - vv * wu) / D;
    if (s < 0.0 || s > 1.0)         // I is outside T
        return 0;
    t = (uv * wu - uu * wv) / D;
    if (t < 0.0 || (s + t) > 1.0)  // I is outside T
        return 0;

    return 1;                       // I is in T
}

//**************************************************************************

Meix::Meix(std::string file_name)
{
  SurfArea=0.f; //initialise

  //Function that reads an STL file ands stores all the triangles in a data structure
  //adapted from: http://stackoverflow.com/questions/22100662/using-ifstream-to-read-floats

  string line;
  fstream myfile;

  myfile.open(file_name, std::fstream::in);

  //if(myfile.fail())  return 0;

  std::vector<string> strings;

  std::list<Triangle> f_list;
  //facets=new list<Facet>();

  Point current_V0;
  Point current_V1;
  Point current_V2;
  Point current_N;

  if (myfile.is_open())
  {
    int global_counter=0; //keep track of file line
    int triangle_counter=0; //keept tranck of triangle index
    int inside_counter=0; //keep track of line within each triangle
    while ( getline (myfile,line) )
    {

      global_counter++;

      strtk::remove_leading_trailing(whitespace, line);

      if(global_counter==1)
      {
        //std::cout << "this is the first line: "<< line << std::endl;
        continue;
      }

      if( strtk::parse(line, whitespace, strings) )
      {
          //  std::cout <<"line: "<<global_counter<< " succeed" << std::endl;
           // strings contains all the values on the in line as strings
      }

      if(strings[0]=="endsolid")
      {
        // std::cout <<"line: "<<global_counter<< " LAST LINE" << std::endl;
        break;
      }

      inside_counter++;

      if(inside_counter==1)
      {
        triangle_counter++;
        //we catch the normal of the triangle
        if(strings[0]!="facet") {std::cout <<global_counter<<" missed the normal "<<strings[0]<< std::endl;return;}
        current_N.x=stof(strings[2]) ; current_N.y=stof(strings[3]) ; current_N.z=stof(strings[4]);

        //x=std::stof(strings[2]) ; y=std::stof(strings[3]) ; z=std::stof(strings[4]);
        // std::cout <<"triangle: "<<triangle_counter<< " normal"<< x <<" "<< y<< " "<<z << std::endl;

      }

      if(inside_counter==3)
      {
        if(strings[0]!="vertex") {std::cout <<global_counter<<" "<<inside_counter<<"missed the vertex"<< std::endl;return;}
        current_V0.x=stof(strings[1]) ; current_V0.y=stof(strings[2]) ; current_V0.z=stof(strings[3]);

        //x=std::stof(strings[1]) ; y=std::stof(strings[2]) ; z=std::stof(strings[3]);
        // std::cout <<"triangle: "<<triangle_counter<< " vertex1 "<< x <<" "<< y<< " "<<z << std::endl;
      }

      if(inside_counter==4)
      {
        if(strings[0]!="vertex") {std::cout <<global_counter<<" "<<inside_counter<<"missed the vertex"<< std::endl;return;}
        current_V1.x=stof(strings[1]) ; current_V1.y=stof(strings[2]) ; current_V1.z=stof(strings[3]);

        //x=std::stof(strings[1]) ; y=std::stof(strings[2]) ; z=std::stof(strings[3]);
        // std::cout <<"triangle: "<<triangle_counter<< " vertex2 "<< x <<" "<< y<< " "<<z << std::endl;
      }

      if(inside_counter==5)
      {
        if(strings[0]!="vertex") {std::cout <<global_counter<<" "<<inside_counter<<"missed the vertex"<< std::endl;return;}
        current_V2.x=stof(strings[1]) ; current_V2.y=stof(strings[2]) ; current_V2.z=stof(strings[3]);

        //x=std::stof(strings[1]) ; y=std::stof(strings[2]) ; z=std::stof(strings[3]);
        // std::cout <<"triangle: "<<triangle_counter<< " vertex3 "<< x <<" "<< y<< " "<<z << std::endl;
      }

      if(inside_counter==7)
      {
        //reset for next triangle
        if(strings[0]!="endfacet") {std::cout <<global_counter<<" "<<inside_counter<<"missed the end"<< std::endl;return;}
        inside_counter=0;

        Triangle current_facet(current_V0, current_V1, current_V2, current_N);

        f_list.push_back(current_facet);

        // std::cout <<"end of facet "<< std::endl;
      }

      strings.clear(); //clear the vector for next line

    }
    myfile.close();
  }

  else std::cout << "Unable to open file";

  //std::cout<<"DONE!"<<std::endl;

  //Convert the list into a vector
  for (std::list<Triangle>::iterator fit = f_list.begin() ; fit != f_list.end() ; fit++)
  {
    Triangle f=*fit;
    Facets.push_back(f);
  }
  CalcSurfArea();

}

//*******************************************************************************************************

void Meix::Rescale(float resc)
{
  for (int i=0 ; i<Facets.size(); i++)
  {
    Facets[i].V0=Facets[i].V0*resc ;
    Facets[i].V1=Facets[i].V1*resc ;
    Facets[i].V2=Facets[i].V2*resc ;
    Facets[i].C=Facets[i].C*resc ;
  }
  CalcSurfArea();
}

//********************************************************************************************************

void Meix::CalcSurfArea()
{
  float global_S=0.f;
  for (int i=0 ; i<Facets.size(); i++)
  {

    Point V0(Facets[i].V0.x,Facets[i].V0.y,Facets[i].V0.z);
    Point V1(Facets[i].V1.x,Facets[i].V1.y,Facets[i].V1.z);
    Point V2(Facets[i].V2.x,Facets[i].V2.y,Facets[i].V2.z);

    Point AB=V0-V1;
    Point BC=V1-V2;
    Point CA=V2-V0;

    float a=sqrt(AB.x*AB.x + AB.y*AB.y + AB.z*AB.z);
    float b=sqrt(BC.x*BC.x + BC.y*BC.y + BC.z*BC.z);
    float c=sqrt(CA.x*CA.x + CA.y*CA.y + CA.z*CA.z);

    float s=0.5f*(a+b+c);

    //S=sqrt(s*(s-a)*(s-b)*(s-c));

    global_S+=sqrt(s*(s-a)*(s-b)*(s-c));

  }
  SurfArea=global_S;
}

//********************************************************************************************************

//Function that checks if a point is inside a closed polyhedron defined by
//a list of facets (or triangles)
//                      List of points to test    , List of facets (mesh)      , List of results , Direction of ray
void Meix::InclusionTest(std::vector<Point>& points , int* inclusion, Point direction)
{

  for (int i=0 ; i<points.size() ; i++)
  {
    Point p_0=points[i];
    Point p_1=p_0+direction;
    Ray R(p_0, p_1);
    int intersection_count=0;
    for (int j=0 ; j<Facets.size() ; j++)
    {
      Point * intersect=new Point(0.0f,0.0f,0.0f);
      int test=intersect3D_RayTriangle( R, Facets[j], intersect );
      if(test>0) intersection_count++;
    }
    if(intersection_count%2==0)
    {
      inclusion[i]=0;
    }
    else
    {
      inclusion[i]=1;
    }
    //std::cout<<"i "<<i<<" inclusion "<<inclusion[i]<<endl;
  }

}

//**********************************************************************************

Meix::~Meix()
{
  Facets.clear(); //that should clean up dynamically allocated memory??
}
