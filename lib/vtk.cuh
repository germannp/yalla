// Write Vtk legacy files, see http://www.vtk.org/wp-content/uploads/
// 2015/04/file-formats.pdf
#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <time.h>


class VtkOutput {
public:
    VtkOutput(std::string base_name);
    VtkOutput(std::string base_name, int N_TIME_STEPS);
    VtkOutput(std::string base_name, int N_TIME_STEPS, int SKIP_STEPS);
    ~VtkOutput(void);
    template<typename Pt> void write_positions(int n_cells, Pt X[]);
    template<typename Field> void write_field(int n_cells,
        const char* data_name, Field f[]);
    void write_connections(int n_connections, int connections[][2]);
private:
    int mN_TIME_STEPS = 0;
    int mSKIP_STEPS = 1;
    int mTimeStep = 0;
    std::string mBASE_NAME;
    std::string mCurrentFile;
    bool mWrite = 0;
    time_t mStart;
};


VtkOutput::VtkOutput(std::string base_name) {
    mBASE_NAME = base_name;
    mkdir("output", 755);
    time(&mStart);
}

VtkOutput::VtkOutput(std::string base_name, int N_TIME_STEPS) {
    mBASE_NAME = base_name;
    mN_TIME_STEPS = N_TIME_STEPS;
    mkdir("output", 755);
    time(&mStart);
}

VtkOutput::VtkOutput(std::string base_name, int N_TIME_STEPS, int SKIP_STEPS) {
    mBASE_NAME = base_name;
    mN_TIME_STEPS = N_TIME_STEPS;
    mSKIP_STEPS = SKIP_STEPS;
    mkdir("output", 755);
    time(&mStart);
}


VtkOutput::~VtkOutput() {
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
}


template<typename Pt> void VtkOutput::write_positions(int n_cells, Pt X[]) {
    if (mTimeStep % mSKIP_STEPS == 0) {
        std::cout << "Integrating " << mBASE_NAME << ", ";
        if (mN_TIME_STEPS > 0) {
            std::cout << std::setw(3)
                << (int)(100.0*mTimeStep/mN_TIME_STEPS) << "% done\r";
        } else {
            std::cout << mTimeStep << " steps done\r";
        }
        std::cout.flush();

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

        mWrite = 1;
    } else {
        mWrite = 0;
    }
    mTimeStep += 1;
}

template<typename Field> void VtkOutput::write_field(int n_cells,
    const char* data_name, Field f[]) {
    if (mWrite) {
        std::ofstream file(mCurrentFile, std::ios_base::app);
        assert(file.is_open());

        file << "\nPOINT_DATA " << n_cells << "\n";
        file << "SCALARS " << data_name << " int\n";
        file << "LOOKUP_TABLE default\n";
        for (int i = 0; i < n_cells; i++)
            file << f[i] << "\n";
    }
}

void VtkOutput::write_connections(int n_connections, int connections[][2]) {
    if (mWrite) {
        std::ofstream file(mCurrentFile, std::ios_base::app);
        assert(file.is_open());

        file << "\nLINES " << n_connections << " " << 3*n_connections << "\n";
        for (int i = 0; i < n_connections; i++)
            file << "2 " << connections[i][0] << " " << connections[i][1] << "\n";
    }
}
