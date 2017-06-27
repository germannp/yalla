#include <fstream>
#include <list>
#include <vector>
#include <string>
#include <cassert>
#include "meix.h"
#include "strtk.hpp"   // http://www.partow.net/programming/strtk

using namespace std;

const char *whitespace    = " \t\r\n\f";

//Overloaded operators for Point
bool operator == (Point& a, Point& b) {
    if(a.x == b.x && a.y == b.y && a.z == b.z) {
        return true;
    }
    else {
        return false;
    }
}

bool operator != (Point& a, Point& b) {
    if(a.x != b.x && a.y != b.y && a.z != b.z)
    {
        return true;
    }
    else {
        return false;
    }
}

Point operator + (Point a, Point b) {
    Point p;
    p.x=a.x + b.x;
    p.y=a.y + b.y;
    p.z=a.z + b.z;
    return p;
}

Point operator - (Point a, Point b) {
    Point p;
    p.x=a.x - b.x;
    p.y=a.y - b.y;
    p.z=a.z - b.z;
    return p;
}

Point operator * (Point a, float s) {
    Point p;
    p.x=a.x * s;
    p.y=a.y * s;
    p.z=a.z * s;
    return p;
}

//******************************************************************

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

#define SMALL_NUM   0.00000001 // anything that avoids division overflow
// dot product (3D) which allows vector operations in arguments
#define dot(u,v)   ((u).x * (v).x + (u).y * (v).y + (u).z * (v).z)

int intersect3D_RayTriangle( Ray R, Triangle T, Point* I ) {
    Point u, v, n;              // triangle vectors
    Point dir, w0, w;           // ray vectors
    float r, a, b;              // params to calc ray-plane intersect

    // get triangle edge vectors and plane normal
    u = T.V1 - T.V0;
    v = T.V2 - T.V0;
    n=T.N;
    if (n.x == 0.0f && n.y==0.0f && n.z==0.0f)    // triangle is degenerate
        return -1;                                // do not deal with this case

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
    float uu, uv, vv, wu, wv, D;
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

Meix::Meix() {
    surf_area=0.f;
    n=0;
}

//**************************************************************************

Meix::Meix(std::string file_name) {
    surf_area=0.f; //initialise

    //Function that reads an STL file ands stores all the triangles in a data structure
    //adapted from: http://stackoverflow.com/questions/22100662/using-ifstream-to-read-floats

    string line;
    fstream myfile;

    myfile.open(file_name, std::fstream::in);

    std::vector<string> strings;

    std::list<Triangle> f_list;

    Point current_V0;
    Point current_V1;
    Point current_V2;
    Point current_N;

    if (myfile.is_open()) {
        int global_counter=0; //keep track of file line
        int triangle_counter=0; //keept tranck of triangle index
        int inside_counter=0; //keep track of line within each triangle
        while ( getline (myfile,line) ) {
            global_counter++;

            strtk::remove_leading_trailing(whitespace, line);

            if(global_counter==1) {
                //std::cout << "this is the first line: "<< line << std::endl;
                continue;
            }

            if( strtk::parse(line, whitespace, strings) )
                // std::cout <<"line: "<<global_counter<< " succeed" << std::endl;

            if(strings[0]=="endsolid") {
                // std::cout <<"line: "<<global_counter<< " LAST LINE" << std::endl;
                break;
            }

            inside_counter++;

            if(inside_counter==1) {
                triangle_counter++;
                //we catch the normal of the triangle
                if(strings[0]!="facet") {std::cout <<global_counter<<" missed the normal "<<strings[0]<< std::endl;return;}
                current_N.x=stof(strings[2]) ; current_N.y=stof(strings[3]) ; current_N.z=stof(strings[4]);
            }

            if(inside_counter==3) {
                if(strings[0]!="vertex") {std::cout <<global_counter<<" "<<inside_counter<<"missed the vertex"<< std::endl;return;}
                current_V0.x=stof(strings[1]) ; current_V0.y=stof(strings[2]) ; current_V0.z=stof(strings[3]);
            }

            if(inside_counter==4) {
                if(strings[0]!="vertex") {std::cout <<global_counter<<" "<<inside_counter<<"missed the vertex"<< std::endl;return;}
                current_V1.x=stof(strings[1]) ; current_V1.y=stof(strings[2]) ; current_V1.z=stof(strings[3]);
            }

            if(inside_counter==5) {
                if(strings[0]!="vertex") {std::cout <<global_counter<<" "<<inside_counter<<"missed the vertex"<< std::endl;return;}
                current_V2.x=stof(strings[1]) ; current_V2.y=stof(strings[2]) ; current_V2.z=stof(strings[3]);
            }

            if(inside_counter==7) {
                //reset for next triangle
                if(strings[0]!="endfacet") {std::cout <<global_counter<<" "<<inside_counter<<"missed the end"<< std::endl;return;}
                inside_counter=0;
                Triangle current_facet(current_V0, current_V1, current_V2, current_N);

                f_list.push_back(current_facet);
            }

            strings.clear(); //clear the vector for next line

        }
        myfile.close();
    }

    else std::cout << "Unable to open file";

    //Convert the list into a vector
    for (std::list<Triangle>::iterator fit = f_list.begin() ; fit != f_list.end() ; fit++) {
        Triangle f=*fit;
        Facets.push_back(f);
    }

    n=Facets.size();
    Calc_surf_area();
}

//*******************************************************************************************************

void Meix::Rescale_relative(float resc) {
    for (int i=0 ; i<Facets.size(); i++) {
        Facets[i].V0=Facets[i].V0*resc ;
        Facets[i].V1=Facets[i].V1*resc ;
        Facets[i].V2=Facets[i].V2*resc ;
        Facets[i].C=Facets[i].C*resc ;
    }
    Calc_surf_area();
}

//*******************************************************************************************************

void Meix::Rescale_absolute(float l) {
    Point r;
    float d,resc;
    for (int i=0 ; i<Facets.size() ; i++) {
        //V0
        r=Facets[i].V0;
        d=sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
        resc=(d+l)/d;
        Facets[i].V0=Facets[i].V0*resc;
        //V1
        r=Facets[i].V1;
        d=sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
        resc=(d+l)/d;
        Facets[i].V1=Facets[i].V1*resc;
        //V2
        r=Facets[i].V2;
        d=sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
        resc=(d+l)/d;
        Facets[i].V2=Facets[i].V2*resc;
        //C
        r=Facets[i].C;
        d=sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
        resc=(d+l)/d;
        Facets[i].C=Facets[i].C*resc;

        //recalculate normal
        Point v=Facets[i].V2-Facets[i].V0;
        Point u=Facets[i].V1-Facets[i].V0;
        Point n(u.y*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.y*v.x);
        float d=sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        Facets[i].N=n*(1.f/d);
    }
    Calc_surf_area();
}

//********************************************************************************************************

//rotation around z axis and around y axis
void Meix::Rotate(float correction_theta, float correction_phi) {
    //rotation around z axis (theta correction)
    // std::cout<<"normal of facet 0 before theta rotation: "<<Facets[0].N.x<<" "<<Facets[0].N.y<<" "<<Facets[0].N.z<<std::endl;
    for(int i=0 ; i<n ; i++) {
        Point old=Facets[i].V0;
        Facets[i].V0.x=old.x*cos(correction_theta)-old.y*sin(correction_theta);
        Facets[i].V0.y=old.x*sin(correction_theta)+old.y*cos(correction_theta);

        old=Facets[i].V1;
        Facets[i].V1.x=old.x*cos(correction_theta)-old.y*sin(correction_theta);
        Facets[i].V1.y=old.x*sin(correction_theta)+old.y*cos(correction_theta);

        old=Facets[i].V2;
        Facets[i].V2.x=old.x*cos(correction_theta)-old.y*sin(correction_theta);
        Facets[i].V2.y=old.x*sin(correction_theta)+old.y*cos(correction_theta);

        old=Facets[i].C;
        Facets[i].C.x=old.x*cos(correction_theta)-old.y*sin(correction_theta);
        Facets[i].C.y=old.x*sin(correction_theta)+old.y*cos(correction_theta);
        //Recalculate normal
        Point v=Facets[i].V2 - Facets[i].V0;
        Point u=Facets[i].V1 - Facets[i].V0;
        Point n(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x);
        float d=sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        Facets[i].N=n*(1.f/d);
    }
    // std::cout<<"normal of facet 0 after theta rotation: "<<Facets[0].N.x<<" "<<Facets[0].N.y<<" "<<Facets[0].N.z<<std::endl;

    //rotation around y axis (phi correction)
    for(int i=0 ; i<n ; i++) {
        Point old=Facets[i].V0;
        Facets[i].V0.x=old.x*cos(correction_phi) - old.z*sin(correction_phi);
        Facets[i].V0.z=old.x*sin(correction_phi) + old.z*cos(correction_phi);

        old=Facets[i].V1;
        Facets[i].V1.x=old.x*cos(correction_phi) - old.z*sin(correction_phi);
        Facets[i].V1.z=old.x*sin(correction_phi) + old.z*cos(correction_phi);

        old=Facets[i].V2;
        Facets[i].V2.x=old.x*cos(correction_phi) - old.z*sin(correction_phi);
        Facets[i].V2.z=old.x*sin(correction_phi) + old.z*cos(correction_phi);

        old=Facets[i].C;
        Facets[i].C.x=old.x*cos(correction_phi) - old.z*sin(correction_phi);
        Facets[i].C.z=old.x*sin(correction_phi) + old.z*cos(correction_phi);
        //Recalculate normal
        Point v=Facets[i].V2 - Facets[i].V0;
        Point u=Facets[i].V1 - Facets[i].V0;
        Point n(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x);
        float d=sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
        Facets[i].N=n*(1.f/d);
    }
    // std::cout<<"normal of facet 0 after phi rotation: "<<Facets[0].N.x<<" "<<Facets[0].N.y<<" "<<Facets[0].N.z<<std::endl;

}

//********************************************************************************************************

void Meix::Translate(Point translation_vector) {
    for (int i=0 ; i<n ; i++) {
        Facets[i].V0=Facets[i].V0 + translation_vector;
        Facets[i].V1=Facets[i].V1 + translation_vector;
        Facets[i].V2=Facets[i].V2 + translation_vector;
        Facets[i].C=Facets[i].C + translation_vector;
    }
}

//********************************************************************************************************

Point Meix::Get_centroid() {
    Point centroid;
    for (int i=0 ; i<n ; i++) {
        centroid=centroid+Facets[i].C;
    }
    centroid=centroid*(1.f/float(n));
    return centroid;
}

//********************************************************************************************************

void Meix::Calc_surf_area() {
    float global_S=0.f;
    for (int i=0 ; i<n; i++) {
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

        global_S+=sqrt(s*(s-a)*(s-b)*(s-c));
    }
    surf_area=global_S;
}

//********************************************************************************************************

//Function that checks if a point is inside a closed polyhedron defined by
//a list of facets (or triangles)
//                      List of points to test    , List of facets (mesh)      , List of results , Direction of ray
void Meix::InclusionTest(std::vector<Point>& points , int* inclusion, Point direction) {
    for (int i=0 ; i<points.size() ; i++) {
        Point p_0=points[i];
        Point p_1=p_0+direction;
        Ray R(p_0, p_1);
        int intersection_count=0;
        for (int j=0 ; j<n ; j++) {
            Point * intersect=new Point(0.0f,0.0f,0.0f);
            int test=intersect3D_RayTriangle( R, Facets[j], intersect );
            if(test>0) intersection_count++;
        }
        if(intersection_count%2==0) {
            inclusion[i]=0;
        }
        else {
            inclusion[i]=1;
        }
    }
}

//**********************************************************************************

//writes the whole meix data structure as a vtk file
void Meix::WriteVtk (std::string output_tag) {
    std::string filename="output/"+output_tag+".meix.vtk";
    std::ofstream meix_file(filename);
    assert(meix_file.is_open());

    meix_file << "# vtk DataFile Version 3.0\n";
    meix_file << output_tag+".meix" << "\n";
    meix_file << "ASCII\n";
    meix_file << "DATASET POLYDATA\n";

    meix_file << "\nPOINTS " << 3*n << " float\n";
    for (auto i = 0; i < n; i++) {
        meix_file <<Facets[i].V0.x << " " << Facets[i].V0.y << " " << Facets[i].V0.z << "\n";
        meix_file <<Facets[i].V1.x << " " << Facets[i].V1.y << " " << Facets[i].V1.z << "\n";
        meix_file <<Facets[i].V2.x << " " << Facets[i].V2.y << " " << Facets[i].V2.z << "\n";
    }

    meix_file << "\nPOLYGONS " << n << " " << 4*n << "\n";
    for (auto i = 0; i < 3*n; i+=3) {
        meix_file << "3 " << i <<" "<<i+1 <<" "<<i+2 << "\n";
    }
    meix_file.close();
}

//**********************************************************************************

Meix::~Meix() {
    Facets.clear(); //that should clean up dynamically allocated memory??
}
