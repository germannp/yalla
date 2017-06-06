#ifndef _MEIX_H_
#define _MEIX_H_

#include <string>
#include <list>
#include <vector>
#include <cmath>
#include <iostream>

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

//overloaded operators
bool operator == (Point& , Point& );
bool operator != (Point& , Point& );
Point operator + (Point , Point );
Point operator - (Point , Point );
Point operator * (Point , float );

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
    Point C;
    Point N;

    Triangle ()
    {
      V0.x= 0.0f; V0.y= 0.0f; V0.z= 0.0f;
      V1.x= 0.0f; V1.y= 0.0f; V1.z= 0.0f;
      V2.x= 0.0f; V2.y= 0.0f; V2.z= 0.0f;
      C.x= 0.0f; C.y= 0.0f; C.z= 0.0f;
      N.x= 0.0f; N.y= 0.0f; N.z= 0.0f;
    }
    Triangle (Point a, Point b, Point c)
    {
      V0= a;
      V1= b;
      V2= c;
      C=(V0+V1+V2)*(1.f/3.f);
      N.x=0.f; N.y=0.f; N.z=0.f;
    }
    Triangle (Point a, Point b, Point c, Point n)
    {
      V0= a;
      V1= b;
      V2= c;
      C=(V0+V1+V2)*(1.f/3.f);
      N= n;
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

// intersect3D_RayTriangle(): find the 3D intersection of a ray with a triangle
//    Input:  a ray R, and a triangle T
//    Output: *I = intersection point (when it exists)
//    Return: -1 = triangle is degenerate (a segment or point)
//             0 =  disjoint (no intersect)
//             1 =  intersect in unique point I1
//             2 =  are in the same plane

int intersect3D_RayTriangle( Ray , Triangle , Point* );

//*****************************************************************************

class Meix
{
  public:
    std::vector<Triangle>Facets;
    float SurfArea;

    Meix(std::string);
    void Rescale(float);
    void CalcSurfArea();
    void InclusionTest(std::vector<Point>& , int* , Point);

    ~Meix();
};

#endif
