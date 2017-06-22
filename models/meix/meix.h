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

    Point() {
        x=0.0f; y=0.0f; z=0.0f;
    }

    Point(float a, float b, float c) {
        x=a; y=b; z=c;
    }
};

//overloaded operators
bool operator == (Point& , Point& );
bool operator != (Point& , Point& );
Point operator + (Point , Point );
Point operator - (Point , Point );
Point operator * (Point , float );

class Ray {
public:
    Point P0;
    Point P1;
    Ray (Point a, Point b) {
        P0= a;
        P1= b;
    }
};

class Plane {
public:
    Point V0;
    Point n;

    Plane (Point a, Point b) {
        V0= a;
        n= b;
    }
};

class Triangle {
public:
    Point V0;
    Point V1;
    Point V2;
    Point C;
    Point N;

    Triangle () {
        V0.x= 0.0f; V0.y= 0.0f; V0.z= 0.0f;
        V1.x= 0.0f; V1.y= 0.0f; V1.z= 0.0f;
        V2.x= 0.0f; V2.y= 0.0f; V2.z= 0.0f;
        C.x= 0.0f; C.y= 0.0f; C.z= 0.0f;
        N.x= 0.0f; N.y= 0.0f; N.z= 0.0f;
        }
    Triangle (Point a, Point b, Point c) {
        V0= a;
        V1= b;
        V2= c;
        C=(V0+V1+V2)*(1.f/3.f);
        N.x=0.f; N.y=0.f; N.z=0.f;
        }
    Triangle (Point a, Point b, Point c, Point n) {
        V0= a;
        V1= b;
        V2= c;
        C=(V0+V1+V2)*(1.f/3.f);
        N= n;
    }
};

int intersect3D_RayTriangle( Ray , Triangle , Point* );

//*****************************************************************************

class Meix {
public:
    std::vector<Triangle>Facets;
    float SurfArea;
    int n;

    Meix();
    Meix(std::string);
    void Rescale_relative(float);
    void Rescale_absolute(float);
    void CalcSurfArea();
    void InclusionTest(std::vector<Point>& , int* , Point);
    void WriteVtk(std::string);

    ~Meix();
};

#endif
