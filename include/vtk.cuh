// Print status and read and write Vtk legacy files, see
// http://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf
#pragma once

#include <assert.h>
#include <sys/stat.h>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

#include "links.cuh"
#include "polarity.cuh"
#include "utils.cuh"


template<typename Pt, template<typename> class Solver>
class Solution;

template<typename Prop>
struct Property;


class Vtk_output {
    int n_points;
    int n_to_write;
    bool* mask = NULL;
    int time_step{0};
    std::string base_name;
    std::string output_dir;
    std::string current_path;
    bool verbose;
    bool point_data_started;
    time_t t_0;

public:
    // Files are stored as output/base_name_#.vtk
    Vtk_output(std::string base_name, std::string output_path = "output/", bool verbose = true);
    ~Vtk_output(void);
    // Write x, y, and z component of Pt; has to be written first
    template<typename Pt, template<typename> class Solver>
    void write_positions(Solution<Pt, Solver>& points, bool* input_mask = NULL);
    // Write links, see links.cuh; if written has to be second
    void write_links(Links& links);
    // Write further components of Pt
    template<typename Pt, template<typename> class Solver>
    void write_field(Solution<Pt, Solver>& points, const char* data_name = "w",
        float Pt::*field = &Pt::w);
    // Write a polarity vector of Pt (specify set of spherical coordinates
    // otherwise is set to theta and phi by default), see polarity.cuh.
    // Writes {0, 0, 0} for the default theta = phi = 0.
    template<typename Pt, float Pt::*theta = &Pt::theta, float Pt::*phi = &Pt::phi, template<typename> class Solver>
    void write_polarity(Solution<Pt, Solver>& points, const char* data_name = "polarity");
    // Write not integrated property, see property.cuh
    template<typename Prop>
    void write_property(Property<Prop>& property);
};

Vtk_output::Vtk_output(std::string base_name, std::string output_path, bool verbose)
    : base_name{base_name}, output_dir{output_path}, verbose{verbose}
{
    if(output_dir.back() != '/'){
        output_dir.append("/");
        std::cout<<output_dir<<std::endl;
    }
    mkdir(output_dir.c_str(), 0755);
    time(&t_0);
}

Vtk_output::~Vtk_output()
{
    if (!verbose) return;

    auto t_f = time(NULL);
    auto duration = t_f - t_0;
    std::cout << "Integrating " << base_name << ", ";
    if (duration < 60)
        std::cout << duration << " seconds";
    else if (duration < 60 * 60)
        std::cout << duration / 60 << "m " << duration % 60 << "s";
    else
        std::cout << duration / (60 * 60) << "h " << duration % (60 * 60)
                  << "m";
    std::cout << " taken (" << n_points
              << " points).        \n";  // Overwrite everything
}

template<typename Pt, template<typename> class Solver>
void Vtk_output::write_positions(Solution<Pt, Solver>& points, bool* input_mask)
{
    n_points = *points.h_n;
    mask = input_mask;

    if (mask != NULL) {
        n_to_write = 0;
        for (auto i = 0; i < n_points; i++) n_to_write += 1 * mask[i];
    } else {
        n_to_write = n_points;
    }

    current_path =
        output_dir + base_name + "_" + std::to_string(time_step) + ".vtk";
    std::ofstream file(current_path);
    assert(file.is_open());

    file << "# vtk DataFile Version 3.0\n";
    file << base_name << "\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";
    file << "\nPOINTS " << n_to_write << " float\n";

    for (auto i = 0; i < n_points; i++) {
        if ((mask != NULL) and (mask[i] == 0)) continue;

        file << points.h_X[i].x << " " << points.h_X[i].y << " "
             << points.h_X[i].z << "\n";
    }

    file << "\nVERTICES " << n_to_write << " " << 2 * n_to_write << "\n";
    for (auto i = 0; i < n_to_write; i++) file << "1 " << i << "\n";

    point_data_started = false;
    time_step += 1;
    if (!verbose) return;

    std::cout << "Integrating " << base_name << ", ";
    std::cout << time_step << " steps done (" << n_points
              << " points)        \r";
    std::cout.flush();
}

void Vtk_output::write_links(Links& links)
{
    std::ofstream file(current_path, std::ios_base::app);
    assert(file.is_open());

    file << "\nLINES " << *links.h_n << " " << 3 * *links.h_n << "\n";
    for (auto i = 0; i < *links.h_n; i++)
        file << "2 " << links.h_link[i].a << " " << links.h_link[i].b << "\n";
}

template<typename Pt, template<typename> class Solver>
void Vtk_output::write_field(
    Solution<Pt, Solver>& points, const char* data_name, float Pt::*field)
{
    std::ofstream file(current_path, std::ios_base::app);
    assert(file.is_open());

    if (!point_data_started) {
        file << "\nPOINT_DATA " << n_to_write << "\n";

        point_data_started = true;
    }
    file << "SCALARS " << data_name << " float\n";
    file << "LOOKUP_TABLE default\n";

    for (auto i = 0; i < n_points; i++) {
        if ((mask != NULL) and (mask[i] == 0)) continue;
        file << points.h_X[i].*field << "\n";
    }
}

template<typename Pt, float Pt::*theta, float Pt::*phi, template<typename> class Solver>
void Vtk_output::write_polarity(Solution<Pt, Solver>& points, const char* data_name)
{
    std::ofstream file(current_path, std::ios_base::app);
    assert(file.is_open());

    if (!point_data_started) {
        file << "\nPOINT_DATA " << n_to_write << "\n";

        point_data_started = true;
    }
    file << "NORMALS " << data_name << " float\n";
    for (auto i = 0; i < n_points; i++) {
        if ((mask != NULL) and (mask[i] == 0)) continue;

        auto n = pol_to_float3<Pt, theta, phi>(points.h_X[i]);
        if ((points.h_X[i].*theta == 0) and (points.h_X[i].*phi == 0)) n.z = 0;
        file << n.x << " " << n.y << " " << n.z << "\n";
    }
}

template<typename Prop>
void Vtk_output::write_property(Property<Prop>& property)
{
    std::ofstream file(current_path, std::ios_base::app);
    assert(file.is_open());

    if (!point_data_started) {
        file << "\nPOINT_DATA " << n_to_write << "\n";

        point_data_started = true;
    }

    std::string ptype = typeid(Prop).name();
    if (ptype == "f")
        ptype = "float";
    else
        ptype = "int";

    assert(n_points <= property.n_max);
    file << "SCALARS " << property.name << " " << ptype << "\n";
    file << "LOOKUP_TABLE default\n";
    for (auto i = 0; i < n_points; i++) {
        if ((mask != NULL) and (mask[i] == 0)) continue;
        file << property.h_prop[i] << "\n";
    }
}


class Vtk_input {
    std::string file_name;

public:
    Vtk_input(std::string file_name);
    std::streampos find_entry(std::string, std::string);
    template<typename Pt, template<typename> class Solver>
    void read_positions(Solution<Pt, Solver>& points);
    // Read polarity of Pt, see polarity.cuh
    template<typename Pt, template<typename> class Solver>
    void read_polarity(Solution<Pt, Solver>& points);
    // Read further field of Pt
    template<typename Pt, template<typename> class Solver>
    void read_field(Solution<Pt, Solver>& points, const char* data_name = "w",
        float Pt::*field = &Pt::w);
    // Read property, see property.cuh
    template<typename Prop>
    void read_property(Property<Prop>& property, std::string prop_name);
    int n_points;
};

Vtk_input::Vtk_input(std::string file_name) : file_name{file_name}
{
    std::string line;
    std::ifstream input_file;
    std::vector<std::string> items;

    input_file.open(file_name, std::fstream::in);
    assert(input_file.is_open());

    for (auto i = 0; i < 6; i++) {
        getline(input_file, line);
        items = split(line);
        if (items.size() == 0) continue;
        if (items[0] == "POINTS") {
            n_points = stoi(items[1]);
            items.clear();
            break;
        }
    }
}

std::streampos Vtk_input::find_entry(std::string keyword1, std::string keyword2)
{
    std::ifstream input_file(file_name);
    assert(input_file.is_open());

    input_file.seekg(0);  // Start at beginning of file

    std::string word1 = "";
    std::string word2 = "";
    std::string line;
    std::vector<std::string> items;

    getline(input_file, line);  // Skip header to avoid false matches
    getline(input_file, line);
    getline(input_file, line);
    getline(input_file, line);

    while (word1 != keyword1 or word2 != keyword2) {
        getline(input_file, line);
        items = split(line);
        if (items.size() > 1) {
            word1 = items[0];
            word2 = items[1];
        }
        items.clear();
    }
    return input_file.tellg();
}

template<typename Pt, template<typename> class Solver>
void Vtk_input::read_positions(Solution<Pt, Solver>& points)
{
    std::ifstream input_file(file_name);
    assert(input_file.is_open());

    // Set the read position to the last line read
    input_file.seekg(find_entry("POINTS", std::to_string(n_points)));
    std::string line;
    std::vector<std::string> items;

    // Read coordinates
    for (int i = 0; i < n_points; i++) {
        getline(input_file, line);
        items = split(line);
        points.h_X[i].x = stof(items[0]);
        points.h_X[i].y = stof(items[1]);
        points.h_X[i].z = stof(items[2]);
        items.clear();
    }
}

template<typename Pt, template<typename> class Solver>
void Vtk_input::read_polarity(Solution<Pt, Solver>& points)
{
    std::ifstream input_file(file_name);
    assert(input_file.is_open());

    input_file.seekg(find_entry("NORMALS", "polarity"));

    std::string line;
    std::vector<std::string> items;

    // Read normals
    for (auto i = 0; i < n_points; i++) {
        getline(input_file, line);
        items = split(line);
        items.clear();
        auto x = stof(items[0]);
        auto y = stof(items[1]);
        auto z = stof(items[2]);
        auto dist = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
        if (dist == 0) {
            points.h_X[i].phi = 0.0f;
            points.h_X[i].theta = 0.0f;
        } else {
            points.h_X[i].phi = atan2(y, x);
            points.h_X[i].theta = acos(z);  // The normals are unit vectors
        }
    }
}

template<typename Pt, template<typename> class Solver>
void Vtk_input::read_field(
    Solution<Pt, Solver>& points, const char* data_name, float Pt::*field)
{
    std::ifstream input_file(file_name);
    assert(input_file.is_open());

    input_file.seekg(find_entry("SCALARS", data_name));

    std::string line;
    std::vector<std::string> items;

    getline(input_file, line);  // LOOKUP_TABLE line

    for (int i = 0; i < n_points; i++) {
        getline(input_file, line);
        std::istringstream(line) >> points.h_X[i].*field;
    }
}

template<typename Prop>
void Vtk_input::read_property(Property<Prop>& property, std::string prop_name)
{
    std::ifstream input_file(file_name);
    assert(input_file.is_open());

    input_file.seekg(find_entry("SCALARS", prop_name));

    std::string line;
    std::vector<std::string> items;

    getline(input_file, line);  // LOOKUP_TABLE line

    assert(n_points <= property.n_max);
    for (int i = 0; i < n_points; i++) {
        getline(input_file, line);
        std::istringstream(line) >> property.h_prop[i];
    }
}
