// Write Vtk legacy files, see http://www.vtk.org/wp-content/uploads/
// 2015/04/file-formats.pdf
#include <time.h>
#include <sys/stat.h>
#include <cassert>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>


template<typename Pt, int N_MAX, template<typename, int> class Solver>
class Solution;

template<int N_LINKS_MAX>
struct Protrusions;

template<int N_MAX, typename Prop>
struct Property;


class VtkOutput {
public:
    VtkOutput(std::string base_name, int N_TIME_STEPS, int SKIP_STEPS);
    VtkOutput(std::string base_name, int N_TIME_STEPS) : VtkOutput(base_name, N_TIME_STEPS, 1) {}
    explicit VtkOutput(std::string base_name) : VtkOutput(base_name, 0, 1) {}
    ~VtkOutput(void);
    void print_progress();
    void print_done();
    template<typename Pt, int N_MAX, template<typename, int> class Solver>
    void write_positions(Solution<Pt, N_MAX, Solver>& bolls);
    template<typename Pt, int N_MAX, template<typename, int> class Solver>
    void write_field(Solution<Pt, N_MAX, Solver>& bolls,
        const char* data_name = "w", float Pt::*field = &Pt::w);
    template<typename Pt, int N_MAX, template<typename, int> class Solver>
    void write_polarity(Solution<Pt, N_MAX, Solver>& bolls);
    template<int N_MAX, typename Prop>
    void write_property(Property<N_MAX, Prop>& property);
    template<int N_LINKS_MAX>
    void write_protrusions(Protrusions<N_LINKS_MAX>& links, int n_links = N_LINKS_MAX);

protected:
    int mN_cells;
    int mN_TIME_STEPS;
    int mSKIP_STEPS;
    int mTimeStep = -1;
    std::string mBASE_NAME;
    std::string mCurrentFile;
    bool mWrite;
    bool mPDataStarted;
    bool mDone = false;
    time_t mStart;
};


VtkOutput::VtkOutput(std::string base_name, int N_TIME_STEPS, int SKIP_STEPS) {
    mBASE_NAME = base_name;
    mN_TIME_STEPS = N_TIME_STEPS;
    mSKIP_STEPS = SKIP_STEPS;
    mkdir("output", 755);
    time(&mStart);
}

VtkOutput::~VtkOutput() {
    if (!mDone) print_done();
}


void VtkOutput::print_progress() {
    if (mTimeStep % mSKIP_STEPS == 0) {
        std::cout << "Integrating " << mBASE_NAME << ", ";
        if (mN_TIME_STEPS > 0) {
            std::cout << std::setw(3)
                << static_cast<int>(100.0*(mTimeStep + 1)/mN_TIME_STEPS)
                << "% done\r";
        } else {
            std::cout << mTimeStep + 1 << " steps done\r";
        }
        std::cout.flush();
        mWrite = true;
        mPDataStarted = false;
    } else {
        mWrite = false;
    }
    mTimeStep += 1;
}

void VtkOutput::print_done() {
    auto end = time(NULL);

    auto duration = end - mStart;
    std::cout << "Integrating " << mBASE_NAME << ", ";
    if (duration < 60)
    std::cout << duration << " seconds";
    else if (duration < 60*60)
    std::cout << duration/60 << "m " << duration % 60 << "s";
    else
    std::cout << duration/60/60 << "h " << duration % 60*60 << "m";
    std::cout << " taken." << std::endl;

    mDone = true;
}


template<typename Pt, int N_MAX, template<typename, int> class Solver>
void VtkOutput::write_positions(Solution<Pt, N_MAX, Solver>& bolls) {
    print_progress();
    if (!mWrite) return;

    mN_cells = bolls.get_n();
    assert(mN_cells <= N_MAX);

    mCurrentFile = "output/" + mBASE_NAME + "_" + std::to_string(mTimeStep/mSKIP_STEPS)
        + ".vtk";
    std::ofstream file(mCurrentFile);
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

template<typename Pt, int N_MAX, template<typename, int> class Solver>
void VtkOutput::write_field(Solution<Pt, N_MAX, Solver>& bolls,
        const char* data_name, float Pt::*field) {
    if (!mWrite) return;

    std::ofstream file(mCurrentFile, std::ios_base::app);
    assert(file.is_open());

    auto mN_cells = bolls.get_n();
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
    if (!mWrite) return;

    std::ofstream file(mCurrentFile, std::ios_base::app);
    assert(file.is_open());

    auto mN_cells = bolls.get_n();
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
    if (!mWrite) return;

    std::ofstream file(mCurrentFile, std::ios_base::app);
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

template<int N_LINKS_MAX>
void VtkOutput::write_protrusions(Protrusions<N_LINKS_MAX>& links, int n_links) {
    if (!mWrite) return;

    assert(n_links <= N_LINKS_MAX);

    std::ofstream file(mCurrentFile, std::ios_base::app);
    assert(file.is_open());

    file << "\nLINES " << n_links << " " << 3*n_links << "\n";
    for (auto i = 0; i < n_links; i++)
        file << "2 " << links.h_link[i].a << " " << links.h_link[i].b << "\n";
}
