// Print status and write Vtk legacy files, see http://www.vtk.org/wp-content/
// uploads/2015/04/file-formats.pdf
#pragma once

#include <time.h>
#include <sys/stat.h>
#include <assert.h>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>


template<typename Pt, int n_max, template<typename, int> class Solver>
class Solution;

template<int n_links>
class Links;

template<int n_max, typename Prop>
struct Property;


class Vtk_output {
public:
    // Files are stored as output/base_name_###.vtk
    Vtk_output(std::string base_name);
    ~Vtk_output(void);
    // Write x, y, and z component of Pt; has to be written first
    template<typename Pt, int n_max, template<typename, int> class Solver>
    void write_positions(Solution<Pt, n_max, Solver>& bolls);
    // Write links, see links.cuh; if written has to be second
    template<int n_links>
    void write_links(Links<n_links>& links);
    // Write further components of Pt
    template<typename Pt, int n_max, template<typename, int> class Solver>
    void write_field(Solution<Pt, n_max, Solver>& bolls,
        const char* data_name = "w", float Pt::*field = &Pt::w);
    // Write polarity from phi and theta of Pt, see polarity.cuh
    template<typename Pt, int n_max, template<typename, int> class Solver>
    void write_polarity(Solution<Pt, n_max, Solver>& bolls);
    // Write not integrated property, see property.cuh
    template<int n_max, typename Prop>
    void write_property(Property<n_max, Prop>& property);

protected:
    int n_bolls;
    int time_step = 0;
    std::string base_name;
    std::string current_path;
    bool point_data_started;
    time_t t_0;
};


Vtk_output::Vtk_output(std::string name) {
    base_name = name;
    mkdir("output", 755);
    time(&t_0);
}

Vtk_output::~Vtk_output() {
    auto t_f = time(NULL);
    auto duration = t_f - t_0;
    std::cout << "Integrating " << base_name << ", ";
    if (duration < 60)
        std::cout << duration << " seconds";
    else if (duration < 60*60)
        std::cout << duration/60 << "m " << duration % 60 << "s";
    else
        std::cout << duration/60/60 << "h " << duration % 60*60 << "m";
    std::cout << " taken (" << n_bolls << " bolls).        \n";  // Overwrite everything
}


template<typename Pt, int n_max, template<typename, int> class Solver>
void Vtk_output::write_positions(Solution<Pt, n_max, Solver>& bolls) {
    n_bolls = *bolls.h_n;
    assert(n_bolls <= n_max);

    std::cout << "Integrating " << base_name << ", ";
    std::cout << time_step << " steps done (" << n_bolls << " bolls)        \r";
    std::cout.flush();
    point_data_started = false;
    time_step += 1;

    current_path = "output/" + base_name + "_" + std::to_string(time_step)
        + ".vtk";
    std::ofstream file(current_path);
    assert(file.is_open());

    file << "# vtk DataFile Version 3.0\n";
    file << base_name << "\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";

    file << "\nPOINTS " << n_bolls << " float\n";
    for (auto i = 0; i < n_bolls; i++)
        file << bolls.h_X[i].x << " " << bolls.h_X[i].y << " " << bolls.h_X[i].z << "\n";

    file << "\nVERTICES " << n_bolls << " " << 2*n_bolls << "\n";
    for (auto i = 0; i < n_bolls; i++)
        file << "1 " << i << "\n";
}

template<int n_links>
void Vtk_output::write_links(Links<n_links>& links) {
    std::ofstream file(current_path, std::ios_base::app);
    assert(file.is_open());

    file << "\nLINES " << *links.h_n << " " << 3**links.h_n << "\n";
    for (auto i = 0; i < *links.h_n; i++)
        file << "2 " << links.h_link[i].a << " " << links.h_link[i].b << "\n";
}

template<typename Pt, int n_max, template<typename, int> class Solver>
void Vtk_output::write_field(Solution<Pt, n_max, Solver>& bolls,
        const char* data_name, float Pt::*field) {
    std::ofstream file(current_path, std::ios_base::app);
    assert(file.is_open());

    if (!point_data_started) {
        file << "\nPOINT_DATA " << n_bolls << "\n";
        point_data_started = true;
    }
    file << "SCALARS " << data_name << " float\n";
    file << "LOOKUP_TABLE default\n";
    for (auto i = 0; i < n_bolls; i++)
        file << bolls.h_X[i].*field << "\n";
}

template<typename Pt, int n_max, template<typename, int> class Solver>
void Vtk_output::write_polarity(Solution<Pt, n_max, Solver>& bolls) {
    std::ofstream file(current_path, std::ios_base::app);
    assert(file.is_open());

    if (!point_data_started) {
        file << "\nPOINT_DATA " << n_bolls << "\n";
        point_data_started = true;
    }
    file << "NORMALS polarity float\n";
    for (auto i = 0; i < n_bolls; i++) {
        float3 n {0};
        if ((bolls.h_X[i].phi != 0) or (bolls.h_X[i].theta != 0)) {
            n.x = sinf(bolls.h_X[i].theta)*cosf(bolls.h_X[i].phi);
            n.y = sinf(bolls.h_X[i].theta)*sinf(bolls.h_X[i].phi);
            n.z = cosf(bolls.h_X[i].theta);
        }
        file << n.x << " " << n.y << " " << n.z << "\n";
    }
}

template<int n_max, typename Prop>
void Vtk_output::write_property(Property<n_max, Prop>& property) {
    std::ofstream file(current_path, std::ios_base::app);
    assert(file.is_open());

    if (!point_data_started) {
        file << "\nPOINT_DATA " << n_bolls << "\n";
        point_data_started = true;
    }
    file << "SCALARS " << property.name << " int\n";
    file << "LOOKUP_TABLE default\n";
    for (auto i = 0; i < n_bolls; i++)
        file << property.h_prop[i] << "\n";
}