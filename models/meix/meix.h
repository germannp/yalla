#ifndef _MEIX_H_
#define _MEIX_H_

#include <string>
#include <vector>

using namespace std;

struct Facet
{
  float nx,ny,nz;
  float p1x,p1y,p1z;
  float p2x,p2y,p2z;
  float p3x,p3y,p3z;
  float cx,cy,cz;
  int n;
};

extern const char *whitespace;

//Function that reads an STL file ands stores all the triangles in a data structure
//adapted from: http://stackoverflow.com/questions/22100662/using-ifstream-to-read-floats

void GetMeixSTL(string, std::vector<Facet>&);

void MeixRescale(std::vector<Facet>&, std::vector<Facet>&, float);

#endif
