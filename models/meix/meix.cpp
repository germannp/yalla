#include <fstream>
#include <list>
#include <vector>
#include "meix.h"
#include "strtk.hpp"   // http://www.partow.net/programming/strtk

using namespace std;

const char *whitespace    = " \t\r\n\f";

//Function that reads an STL file ands stores all the triangles in a data structure
//adapted from: http://stackoverflow.com/questions/22100662/using-ifstream-to-read-floats


void MeixRescale(std::vector<Facet>& list0, std::vector<Facet>& list1, float resc)
{
  for (int i=0 ; i<list0.size(); i++)
  {
    Facet f=list0[i];
    f.p1x=f.p1x*resc ; f.p1y=f.p1y*resc ; f.p1z=f.p1z*resc;
    f.p2x=f.p2x*resc ; f.p2y=f.p2y*resc ; f.p2z=f.p2z*resc;
    f.p3x=f.p3x*resc ; f.p3y=f.p3y*resc ; f.p3z=f.p3z*resc;
    f.cx=f.cx*resc ; f.cy=f.cy*resc ; f.cz=f.cz*resc;

    list1.push_back(f);
  }
}



void GetMeixSTL(std::string file_name, std::vector<Facet>& vector)
{

  // float x=0.0f;
  // float y=0.0f;
  // float z=0.0f;

  string line;
  fstream myfile;

  myfile.open(file_name, std::fstream::in);

  //if(myfile.fail())  return 0;

  std::vector<string> strings;

  std::list<Facet> facets;
  //facets=new list<Facet>();

  Facet current_facet;

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

      // if(line=="endsolid GiD")
      // {
      //   // std::cout <<"line: "<<global_counter<< " LAST LINE" << std::endl;
      //   break;
      // }

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

        current_facet.nx=stof(strings[2]) ; current_facet.ny=stof(strings[3]) ; current_facet.nz=stof(strings[4]);

        //x=std::stof(strings[2]) ; y=std::stof(strings[3]) ; z=std::stof(strings[4]);
        // std::cout <<"triangle: "<<triangle_counter<< " normal"<< x <<" "<< y<< " "<<z << std::endl;

      }

      if(inside_counter==3)
      {
        if(strings[0]!="vertex") {std::cout <<global_counter<<" "<<inside_counter<<"missed the vertex"<< std::endl;return;}
        current_facet.p1x=stof(strings[1]) ; current_facet.p1y=stof(strings[2]) ; current_facet.p1z=stof(strings[3]);

        //x=std::stof(strings[1]) ; y=std::stof(strings[2]) ; z=std::stof(strings[3]);
        // std::cout <<"triangle: "<<triangle_counter<< " vertex1 "<< x <<" "<< y<< " "<<z << std::endl;
      }

      if(inside_counter==4)
      {
        if(strings[0]!="vertex") {std::cout <<global_counter<<" "<<inside_counter<<"missed the vertex"<< std::endl;return;}
        current_facet.p2x=stof(strings[1]) ; current_facet.p2y=stof(strings[2]) ; current_facet.p2z=stof(strings[3]);

        //x=std::stof(strings[1]) ; y=std::stof(strings[2]) ; z=std::stof(strings[3]);
        // std::cout <<"triangle: "<<triangle_counter<< " vertex2 "<< x <<" "<< y<< " "<<z << std::endl;
      }

      if(inside_counter==5)
      {
        if(strings[0]!="vertex") {std::cout <<global_counter<<" "<<inside_counter<<"missed the vertex"<< std::endl;return;}
        current_facet.p3x=stof(strings[1]) ; current_facet.p3y=stof(strings[2]) ; current_facet.p3z=stof(strings[3]);

        //x=std::stof(strings[1]) ; y=std::stof(strings[2]) ; z=std::stof(strings[3]);
        // std::cout <<"triangle: "<<triangle_counter<< " vertex3 "<< x <<" "<< y<< " "<<z << std::endl;
      }

      if(inside_counter==7)
      {
        //reset for next triangle
        if(strings[0]!="endfacet") {std::cout <<global_counter<<" "<<inside_counter<<"missed the end"<< std::endl;return;}
        inside_counter=0;

        current_facet.cx=(current_facet.p1x + current_facet.p2x + current_facet.p3x)/3;
        current_facet.cy=(current_facet.p1y + current_facet.p2y + current_facet.p3y)/3;
        current_facet.cz=(current_facet.p1z + current_facet.p2z + current_facet.p3z)/3;

        facets.push_back(current_facet);

        // std::cout <<"end of facet "<< std::endl;
      }

      strings.clear(); //clear the vector for next line

    }
    myfile.close();
  }

  else std::cout << "Unable to open file";

  //std::cout<<"DONE!"<<std::endl;

  //Convert the list into a vector
  for (std::list<Facet>::iterator fit = facets.begin() ; fit != facets.end() ; fit++)
  {
    Facet f=*fit;
    vector.push_back(f);
  }

}
