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


template<typename Pt, int N_MAX, template<typename, int> class Solver>
class Solution;

template<int N_LINKS_MAX>
class Protrusions;

template<int N_MAX, typename Prop>
struct Property;


class VtkOutput {
public:
    // Files are stored as output/base_name_###.vtk
    VtkOutput(std::string base_name);
    ~VtkOutput(void);
    // Write x, y, and z component of Pt; has to be written first
    template<typename Pt, int N_MAX, template<typename, int> class Solver>
    void write_positions(Solution<Pt, N_MAX, Solver>& bolls);
    // Write links, see protrusions.cuh; if written has to be second
    template<int N_LINKS_MAX>
    void write_protrusions(Protrusions<N_LINKS_MAX>& links);
    // Write further components of Pt
    template<typename Pt, int N_MAX, template<typename, int> class Solver>
    void write_field(Solution<Pt, N_MAX, Solver>& bolls,
        const char* data_name = "w", float Pt::*field = &Pt::w);
    // Write polarity from phi and theta of Pt, see epithelium.cuh
    template<typename Pt, int N_MAX, template<typename, int> class Solver>
    void write_polarity(Solution<Pt, N_MAX, Solver>& bolls);
    // Write not integrated property, see property.cuh
    template<int N_MAX, typename Prop>
    void write_property(Property<N_MAX, Prop>& property);

protected:
    int mN_cells;
    int mTimeStep = 0;
    std::string mBASE_NAME;
    std::string mCurrentPath;
    bool mPDataStarted;
    time_t mStart;
};


VtkOutput::VtkOutput(std::string base_name) {
    mBASE_NAME = base_name;
    mkdir("output", 755);
    time(&mStart);
}

VtkOutput::~VtkOutput() {
    auto end = time(NULL);
    auto duration = end - mStart;
    std::cout << "Integrating " << mBASE_NAME << ", ";
    if (duration < 60)
        std::cout << duration << " seconds";
    else if (duration < 60*60)
        std::cout << duration/60 << "m " << duration % 60 << "s";
    else
        std::cout << duration/60/60 << "h " << duration % 60*60 << "m";
    std::cout << " taken.        \n";  // Overwrite everything
}


template<typename Pt, int N_MAX, template<typename, int> class Solver>
void VtkOutput::write_positions(Solution<Pt, N_MAX, Solver>& bolls) {
    std::cout << "Integrating " << mBASE_NAME << ", ";
    std::cout << mTimeStep << " steps done\r";
    std::cout.flush();
    mPDataStarted = false;
    mTimeStep += 1;

    mN_cells = *bolls.h_n;
    assert(mN_cells <= N_MAX);

    mCurrentPath = "output/" + mBASE_NAME + "_" + std::to_string(mTimeStep)
        + ".vtk";
    std::ofstream file(mCurrentPath);
    assert(file.is_open());

    file << "# vtk DataFile Version 3.0\n";
    file << mBASE_NAME << "\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";

    file << "\nPOINTS " << mN_cells << " float\n";
    for (auto i = 0; i < mN_cells; i++)
        file << bolls.h_X[i].x << " " << bolls.h_X[i].y << " " << bolls.h_X[i].z << "\n";

    file << "\nVERTICES " << mN_cells << " " << 2*mN_cells << "\n";
    for (auto i = 0; i < mN_cells; i++)
        file << "1 " << i << "\n";
}

template<int N_LINKS_MAX>
void VtkOutput::write_protrusions(Protrusions<N_LINKS_MAX>& links) {
    std::ofstream file(mCurrentPath, std::ios_base::app);
    assert(file.is_open());

    file << "\nLINES " << *links.h_n << " " << 3**links.h_n << "\n";
    for (auto i = 0; i < *links.h_n; i++)
        file << "2 " << links.h_link[i].a << " " << links.h_link[i].b << "\n";
}

template<typename Pt, int N_MAX, template<typename, int> class Solver>
void VtkOutput::write_field(Solution<Pt, N_MAX, Solver>& bolls,
        const char* data_name, float Pt::*field) {
    std::ofstream file(mCurrentPath, std::ios_base::app);
    assert(file.is_open());

    if (!mPDataStarted) {
        file << "\nPOINT_DATA " << mN_cells << "\n";
        mPDataStarted = true;
    }
    file << "SCALARS " << data_name << " float\n";
    file << "LOOKUP_TABLE default\n";
    for (auto i = 0; i < mN_cells; i++)
        file << bolls.h_X[i].*field << "\n";
}

template<typename Pt, int N_MAX, template<typename, int> class Solver>
void VtkOutput::write_polarity(Solution<Pt, N_MAX, Solver>& bolls) {
    std::ofstream file(mCurrentPath, std::ios_base::app);
    assert(file.is_open());

    if (!mPDataStarted) {
        file << "\nPOINT_DATA " << mN_cells << "\n";
        mPDataStarted = true;
    }
    file << "NORMALS polarity float\n";
    for (auto i = 0; i < mN_cells; i++) {
        float3 n {0};
        if ((bolls.h_X[i].phi != 0) and (bolls.h_X[i].theta != 0)) {
            n.x = sinf(bolls.h_X[i].theta)*cosf(bolls.h_X[i].phi);
            n.y = sinf(bolls.h_X[i].theta)*sinf(bolls.h_X[i].phi);
            n.z = cosf(bolls.h_X[i].theta);
        }
        file << n.x << " " << n.y << " " << n.z << "\n";
    }
}

template<int N_MAX, typename Prop>
void VtkOutput::write_property(Property<N_MAX, Prop>& property) {
    std::ofstream file(mCurrentPath, std::ios_base::app);
    assert(file.is_open());

    if (!mPDataStarted) {
        file << "\nPOINT_DATA " << mN_cells << "\n";
        mPDataStarted = true;
    }
    file << "SCALARS " << property.name << " int\n";
    file << "LOOKUP_TABLE default\n";
    for (auto i = 0; i < mN_cells; i++)
        file << property.h_prop[i] << "\n";
}
