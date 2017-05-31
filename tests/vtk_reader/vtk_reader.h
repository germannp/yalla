//VTK reader class. Reads bolls-generated vtk files and loads the data on a
//bolls solution class.
//Test version, eventually to be included in main IO header vtk.cuh

#ifndef _VTK_IN_
#define _VTK_IN_

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <iterator>
#include <assert.h>

template<typename Pt, int n_max, template<typename, int> class Solver>
class Solution;

template<int n_links>
class Links;

template<int n_max, typename Prop>
struct Property;


//function to split a string by whitespace
//adapted from: https://stackoverflow.com/questions/236129/split-a-string-in-c
template<typename Out>
void split(const std::string &s, char delim, Out result) {
    if(s.length()==0) return;
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}


//A custom class that reads vtk format files written by the bolls framework
//We use a string parser code adapted from:
// http://stackoverflow.com/questions/22100662/using-ifstream-to-read-floats

class Vtk_input {
public:
    Vtk_input(std::string, int& ); //pass the file name to read, returns an int with the number of bolls
    // Read x, y, and z component of Pt; has to be written first
    template<typename Pt, int n_max, template<typename, int> class Solver>
    void read_positions(Solution<Pt, n_max, Solver>& bolls);
    // Read polarity from phi and theta of Pt, see polarity.cuh
    template<typename Pt, int n_max, template<typename, int> class Solver>
    void read_polarity(Solution<Pt, n_max, Solver>& bolls);
    // Read not integrated property, see property.cuh
    // template<int n_max, typename Prop>
    // void read_property_int(Property<n_max, Prop>& property);
    // template<int n_max, typename Prop>
    // void read_property_float(Property<n_max, Prop>& property);
    template<int n_max, typename Prop>
    void read_property(Property<n_max, Prop>& property);

protected:
    int n_bolls;
    std::streampos bookmark;
    std::string file_name;
};

Vtk_input::Vtk_input(std::string s, int& n) {

    file_name=s;

    std::string line;
    std::ifstream input_file;
    std::vector<std::string> items;


    input_file.open(file_name, std::fstream::in);

    //Line 1
    getline (input_file,line);
    split(line, ' ', std::back_inserter(items));
    // std::cout<<"line 1 "<<items[0]<<std::endl;
    items.clear();

    //Line 2
    getline (input_file,line);
    split(line, ' ', std::back_inserter(items));
    // std::cout<<"line 2 "<<items[0]<<std::endl;
    items.clear();

    //Line 3
    getline (input_file,line);
    split(line, ' ', std::back_inserter(items));
    // std::cout<<"line 3 "<<items[0]<<std::endl;
    items.clear();

    //Line 4
    getline (input_file,line);
    split(line, ' ', std::back_inserter(items));
    // std::cout<<"line 4 "<<items[0]<<std::endl;
    items.clear();

    //Line 5
    getline (input_file,line);
    //split(line, ' ', std::back_inserter(items));
    //std::cout<<"line 5 "<<items[0]<<std::endl;
    //items.clear();

    //Line 6
    getline (input_file,line);
    split(line, ' ', std::back_inserter(items));
    // std::cout<<"line 6 "<<items[0]<<std::endl;
    n_bolls=stoi(items[1]);
    n=n_bolls;
    // std::cout<<"n= "<<n<<std::endl;
    items.clear();

    bookmark = input_file.tellg(); //we save the read position for late read functions


}


template<typename Pt, int n_max, template<typename, int> class Solver>
void Vtk_input::read_positions(Solution<Pt, n_max, Solver>& bolls) {
    //n_bolls = *bolls.h_n;
    //assert(n_bolls <= n_max);

    std::ifstream input_file(file_name);
    assert(input_file.is_open());

    //set the read position to the last line read
    input_file.seekg(bookmark);

    std::string line;
    std::vector<std::string> items;

    //read the list of point coordinates
    for (int i=0 ; i<n_bolls ; i++)
    {
      getline (input_file,line);
      split(line, ' ', std::back_inserter(items));
      bolls.h_X[i].x=stof(items[0]) ; bolls.h_X[i].y=stof(items[1]) ; bolls.h_X[i].z=stof(items[2]) ;
      items.clear();
      // std::cout<<i<<" xyz "<< bolls.h_X[i].x << " "<< bolls.h_X[i].y << " "<< bolls.h_X[i].z <<std::endl;
    }

    //interlude
    getline (input_file,line); //blank line
    getline (input_file,line);
    split(line, ' ', std::back_inserter(items));
    // std::cout<<"vertex list starts "<<items[0]<<std::endl;

    //read the list of VERTICES
    //that's not useful info actually, just for the sake of
    //being compatible with VTK format
    for (int i=0 ; i<n_bolls ; i++)
    {
      getline (input_file,line);
    }

    bookmark=input_file.tellg();

}


template<typename Pt, int n_max, template<typename, int> class Solver>
void Vtk_input::read_polarity(Solution<Pt, n_max, Solver>& bolls) {
    //n_bolls = *bolls.h_n;
    //assert(n_bolls <= n_max);

    std::ifstream input_file(file_name);
    assert(input_file.is_open());

    //set the read position to the last line read
    input_file.seekg(bookmark);

    std::string line;
    std::vector<std::string> items;

    getline (input_file,line); //blank line after positions data
    getline (input_file,line);
    split(line, ' ', std::back_inserter(items));
    // std::cout<<"headline1 "<<items[0]<<std::endl;
    items.clear();
    getline (input_file,line);
    split(line, ' ', std::back_inserter(items));
    // std::cout<<"headline2 "<<items[0]<<std::endl;
    items.clear();

    //read the list of normals
    float x,y,z,d;
    for (int i=0 ; i<n_bolls ; i++)
    {
      getline (input_file,line);
      split(line, ' ', std::back_inserter(items));
      x=stof(items[0]) ; y=stof(items[1]) ; z=stof(items[2]) ;
      items.clear();
      // std::cout<<i<<" xyz "<< x << " "<< y << " "<< z <<std::endl;
      d=sqrt(pow(x,2) + pow(y,2) + pow(z,2));
      if(d==0)
      {
        bolls.h_X[i].phi = 0.0f ; bolls.h_X[i].theta = 0.0f ;
      }
      else
      {
        bolls.h_X[i].phi = atan2(y,x);
        bolls.h_X[i].theta = acos(z); //Assuming the normals are unit vectors, so no need to divide by length
      }
      // std::cout<<i<<" phi "<< bolls.h_X[i].phi << " theta "<< bolls.h_X[i].theta <<std::endl;

    }

    bookmark=input_file.tellg();

}

// template<int n_max, typename Prop>
// void Vtk_input::read_property_int(Property<n_max, Prop>& property) {
//
//   std::ifstream input_file(file_name);
//   assert(input_file.is_open());
//
//   //set the read position to the last line read
//   input_file.seekg(bookmark);
//
//   std::string line;
//   std::vector<std::string> items;
//   getline (input_file,line); //Property header line
//   split(line, ' ', std::back_inserter(items));
//   std::cout<<"headline1 "<<items[0]<<std::endl;
//   items.clear();
//
//   getline (input_file,line); //Lookup table
//   split(line, ' ', std::back_inserter(items));
//   std::cout<<"headline2 "<<items[0]<<std::endl;
//   items.clear();
//
//   for (int i=0 ; i<n_bolls ; i++)
//   {
//     getline (input_file,line);
//     split(line, ' ', std::back_inserter(items));
//     property.h_prop[i]=stoi(items[0]) ;
//     items.clear();
//     std::cout<<i<<" prop "<< property.h_prop[i] <<std::endl;
//   }
//
//     bookmark=input_file.tellg();
// }
//
// template<int n_max, typename Prop>
// void Vtk_input::read_property_float(Property<n_max, Prop>& property) {
//
//   std::ifstream input_file(file_name);
//   assert(input_file.is_open());
//
//   //set the read position to the last line read
//   input_file.seekg(bookmark);
//
//   std::string line;
//   std::vector<std::string> items;
//   getline (input_file,line); //Property header line
//   split(line, ' ', std::back_inserter(items));
//   std::cout<<"headline1 "<<items[0]<<std::endl;
//   items.clear();
//
//   getline (input_file,line); //Lookup table
//   split(line, ' ', std::back_inserter(items));
//   std::cout<<"headline2 "<<items[0]<<std::endl;
//   items.clear();
//
//   for (int i=0 ; i<n_bolls ; i++)
//   {
//     getline (input_file,line);
//     split(line, ' ', std::back_inserter(items));
//     property.h_prop[i]=stof(items[0]) ;
//     items.clear();
//     std::cout<<i<<" prop "<< property.h_prop[i] <<std::endl;
//   }
//
//     bookmark=input_file.tellg();
// }


template<int n_max, typename Prop>
void Vtk_input::read_property(Property<n_max, Prop>& property) {

  std::ifstream input_file(file_name);
  assert(input_file.is_open());

  //set the read position to the last line read
  input_file.seekg(bookmark);

  std::string line;
  std::vector<std::string> items;
  getline (input_file,line); //Property header line
  split(line, ' ', std::back_inserter(items));
  std::cout<<"headline1 "<<items[0]<<std::endl;
  items.clear();

  getline (input_file,line); //Lookup table
  split(line, ' ', std::back_inserter(items));
  std::cout<<"headline2 "<<items[0]<<std::endl;
  items.clear();

  for (int i=0 ; i<n_bolls ; i++)
  {
    getline (input_file,line);
    //split(line, ' ', std::back_inserter(items));
    std::istringstream (line) >> property.h_prop[i] ;
    //property.h_prop[i]=stoi(items[0]) ;
    //items.clear();
    std::cout<<i<<" prop "<< property.h_prop[i] <<std::endl;
  }

    bookmark=input_file.tellg();
}

#endif
