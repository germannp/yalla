// Write Vtk legacy files, see http://www.vtk.org/wp-content/uploads/
// 2015/04/file-formats.pdf
#include <time.h>
#include <sys/stat.h>
#include <cassert>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>


template<typename Pt, int N_MAX, template<typename, int> class Solver>
class Solution;


class VtkOutput {
 public:
    VtkOutput(std::string base_name, int N_TIME_STEPS, int SKIP_STEPS);
    VtkOutput(std::string base_name, int N_TIME_STEPS) : VtkOutput(base_name, N_TIME_STEPS, 1) {}
    explicit VtkOutput(std::string base_name) : VtkOutput(base_name, 0, 1) {}
    ~VtkOutput(void);
    void print_progress();
    void print_done();
    template<typename Pt, int N_MAX, template<typename, int> class Solver>
    void write_positions(Solution<Pt, N_MAX, Solver>& X, int n_cells = N_MAX);
    template<typename Pt, int N_MAX, template<typename, int> class Solver>
    void write_field(Solution<Pt, N_MAX, Solver>& X, int n_cells = N_MAX,
        const char* data_name = "w", float Pt::*field = &Pt::w);
    template<typename Pt, int N_MAX, template<typename, int> class Solver>
    void write_polarity(Solution<Pt, N_MAX, Solver>& X, int n_cells = N_MAX);
    template<typename TYPES>
    void write_type(TYPES type[], int n_cells);
    void write_connections(int connections[][2], int n_connections);

 protected:
    int mN_TIME_STEPS;
    int mSKIP_STEPS;
    int mTimeStep = -1;
    std::string mBASE_NAME;
    std::string mCurrentFile;
    bool mWrite;
    bool mPDataStarted;
    bool mDone = 0;
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
        mWrite = 1;
        mPDataStarted = 0;
    } else {
        mWrite = 0;
    }
    mTimeStep += 1;
}

void VtkOutput::print_done() {
    time_t end = time(NULL), duration;

    duration = end - mStart;
    std::cout << "Integrating " << mBASE_NAME << ", ";
    if (duration < 60)
    std::cout << duration << " seconds";
    else if (duration < 60*60)
    std::cout << duration/60 << "m " << duration % 60 << "s";
    else
    std::cout << duration/60/60 << "h " << duration % 60*60 << "m";
    std::cout << " taken." << std::endl;

    mDone = 1;
}

template<typename Pt, int N_MAX, template<typename, int> class Solver>
void VtkOutput::write_positions(Solution<Pt, N_MAX, Solver>& X, int n_cells) {
    assert(n_cells <= N_MAX);
    print_progress();

    if (!mWrite) return;

    std::stringstream file_name;
    file_name << "output/" << mBASE_NAME << "_" << mTimeStep/mSKIP_STEPS << ".vtk";
    mCurrentFile = file_name.str();
    std::ofstream file(mCurrentFile);
    assert(file.is_open());

    file << "# vtk DataFile Version 3.0\n";
    file << mBASE_NAME << "\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";

    file << "\nPOINTS " << n_cells << " float\n";
    for (int i = 0; i < n_cells; i++)
        file << X[i].x << " " << X[i].y << " " << X[i].z << "\n";

    file << "\nVERTICES " << n_cells << " " << 2*n_cells << "\n";
    for (int i = 0; i < n_cells; i++)
        file << "1 " << i << "\n";
}

template<typename Pt, int N_MAX, template<typename, int> class Solver>
void VtkOutput::write_field(Solution<Pt, N_MAX, Solver>& X, int n_cells,
        const char* data_name, float Pt::*field) {
    if (!mWrite) return;

    assert(n_cells <= N_MAX);

    std::ofstream file(mCurrentFile, std::ios_base::app);
    assert(file.is_open());

    if (!mPDataStarted) {
        file << "\nPOINT_DATA " << n_cells << "\n";
        mPDataStarted = 1;
    }
    file << "SCALARS " << data_name << " float\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < n_cells; i++)
        file << X[i].*field << "\n";
}

template<typename Pt, int N_MAX, template<typename, int> class Solver>
void VtkOutput::write_polarity(Solution<Pt, N_MAX, Solver>& X, int n_cells) {
    if (!mWrite) return;

    assert(n_cells <= N_MAX);

    std::ofstream file(mCurrentFile, std::ios_base::app);
    assert(file.is_open());

    if (!mPDataStarted) {
        file << "\nPOINT_DATA " << n_cells << "\n";
        mPDataStarted = 1;
    }
    file << "NORMALS polarity float\n";
    float3 n;
    for (int i = 0; i < n_cells; i++) {
        n = {0.0f, 0.0f, 0.0f};
        if ((X[i].phi != 0) and (X[i].theta != 0)) {
            n.x = sinf(X[i].theta)*cosf(X[i].phi);
            n.y = sinf(X[i].theta)*sinf(X[i].phi);
            n.z = cosf(X[i].theta);
        }
        file << n.x << " " << n.y << " " << n.z << "\n";
    }
}

template<typename TYPES>
void VtkOutput::write_type(TYPES type[], int n_cells) {
    if (!mWrite) return;

    std::ofstream file(mCurrentFile, std::ios_base::app);
    assert(file.is_open());

    if (!mPDataStarted) {
        file << "\nPOINT_DATA " << n_cells << "\n";
        mPDataStarted = 1;
    }
    file << "SCALARS cell_type int\n";
    file << "LOOKUP_TABLE default\n";
    for (int i = 0; i < n_cells; i++)
        file << type[i] << "\n";
}

void VtkOutput::write_connections(int connections[][2], int n_connections) {
    if (!mWrite) return;

    std::ofstream file(mCurrentFile, std::ios_base::app);
    assert(file.is_open());

    file << "\nLINES " << n_connections << " " << 3*n_connections << "\n";
    for (int i = 0; i < n_connections; i++)
        file << "2 " << connections[i][0] << " " << connections[i][1] << "\n";
}
