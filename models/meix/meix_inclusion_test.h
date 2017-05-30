//Function that performs the Ray-to-triangle inclusion test for a list of points

#ifndef _INC_TEST_H_
#define _INC_TEST_H_

#include <string>
#include <list>
#include <vector>
#include <cmath>
#include <iostream>

#include "meix.h"

class Point
{
  public:
    float x;
    float y;
    float z;

    Point()
    {
      x=0.0f; y=0.0f; z=0.0f;
    }

    Point(float a, float b, float c)
    {
      x=a; y=b; z=c;
    }

    //~Point();
};

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
//////////////////////////////

class Ray
{
  public:
    Point P0;
    Point P1;
    Ray (Point a, Point b)
    {
      P0= a;
      P1= b;
    }
    //~Ray();
};

class Plane
{
  public:
    Point V0;
    Point n;

    Plane (Point a, Point b)
    {
      V0= a;
      n= b;
    }
    //~Plane();
};

class Triangle
{
  public:
    Point V0;
    Point V1;
    Point V2;

    Triangle ()
    {
      V0.x= 0.0f; V0.y= 0.0f; V0.z= 0.0f;
      V1.x= 0.0f; V1.y= 0.0f; V1.z= 0.0f;
      V2.x= 0.0f; V2.y= 0.0f; V2.z= 0.0f;
    }
    Triangle (Point a, Point b, Point c)
    {
      V0= a;
      V1= b;
      V2= c;
    }
    //~Triangle();
};


//The following function checks if a ray intersects with a triangle
// Theory and algorithm: http://geomalgorithms.com/a06-_intersect-2.html

// Copyright 2001 softSurfer, 2012 Dan Sunday
// This code may be freely used and modified for any purpose
// providing that this copyright notice is included with it.
// SoftSurfer makes no warranty for this code, and cannot be held
// liable for any real or imagined damage resulting from its use.
// Users of this code must verify correctness for their application.

#define SMALL_NUM   0.00000001 // anything that avoids division overflow
// dot product (3D) which allows vector operations in arguments
#define dot(u,v)   ((u).x * (v).x + (u).y * (v).y + (u).z * (v).z)



// intersect3D_RayTriangle(): find the 3D intersection of a ray with a triangle
//    Input:  a ray R, and a triangle T
//    Output: *I = intersection point (when it exists)
//    Return: -1 = triangle is degenerate (a segment or point)
//             0 =  disjoint (no intersect)
//             1 =  intersect in unique point I1
//             2 =  are in the same plane
int intersect3D_RayTriangle( Ray R, Facet F, Point* I )
{

    Point    u, v, n;              // triangle vectors
    Point    dir, w0, w;           // ray vectors
    float    r, a, b;              // params to calc ray-plane intersect

    Point p_v0(F.p1x, F.p1y, F.p1z);
    Point p_v1(F.p2x, F.p2y, F.p2z);
    Point p_v2(F.p3x, F.p3y, F.p3z);

    Triangle T(p_v0, p_v1, p_v2);

    // get triangle edge vectors and plane normal
    u = T.V1 - T.V0;
    v = T.V2 - T.V0;
    //n = u * v;              // cross product
    n.x=F.nx ; n.y=F.ny ; n.z=F.nz;
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

//Function that checks if a point is inside a closed polyhedron defined by
//a list of facets (or triangles)
//                      List of points to test    , List of facets (mesh)      , List of results , Direction of ray
void MeixInclusionTest(std::vector<Point>& points , std::vector<Facet>& facets , int* inclusion, Point direction)
{

  for (int i=0 ; i<points.size() ; i++)
  {
    Point p_0=points[i];
    Point p_1=p_0+direction;
    Ray R(p_0, p_1);
    int intersection_count=0;
    for (int j=0 ; j<facets.size() ; j++)
    {
      Point * intersect=new Point(0.0f,0.0f,0.0f);
      int test=intersect3D_RayTriangle( R, facets[j], intersect );
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
#endif
