// Write Vtk legacy files, see http://www.vtk.org/wp-content/uploads/
// 2015/04/file-formats.pdf
#include <cassert>
#include <fstream>
#include <sstream>


class VtkOutput {
public:
    VtkOutput(std::string base_name);
    VtkOutput(std::string base_name, int SKIP_STEPS);
    template<typename Positions> void write_positions(int n_cells, Positions X[]);
    template<typename Field> void write_field(int n_cells,
        const char* data_name, Field f[]);
    void write_connections(int n_connections, int connections[][2]);
private:
    int mSKIP_STEPS = 1;
    int mTimeStep = 0;
    std::string mBASE_NAME;
    std::string mCurrentFile;
    bool mWriteFields = 0;
};

VtkOutput::VtkOutput(std::string base_name) {
    mBASE_NAME = base_name;
    mkdir("output", 755);
}

VtkOutput::VtkOutput(std::string base_name, int SKIP_STEPS) {
    mBASE_NAME = base_name;
    mSKIP_STEPS = SKIP_STEPS;
    mkdir("output", 755);
}

template<typename Positions> void VtkOutput::write_positions(int n_cells, Positions X[]) {
    if (mTimeStep % mSKIP_STEPS == 0) {
        std::stringstream file_name;
        file_name << "output/" << mBASE_NAME << "_" << mTimeStep/mSKIP_STEPS << ".vtk";
        mCurrentFile = file_name.str();
        std::ofstream file(mCurrentFile);
        assert(file.is_open());

        file << "# vtk DataFile Version 3.0\n";
        file << file_name << "\n";
        file << "ASCII\n";
        file << "DATASET POLYDATA\n";

        file << "\nPOINTS " << n_cells << " float\n";
        for (int i = 0; i < n_cells; i++)
            file << X[i].x << " " << X[i].y << " " << X[i].z << "\n";

        file << "\nVERTICES " << n_cells << " " << 2*n_cells << "\n";
        for (int i = 0; i < n_cells; i++)
            file << "1 " << i << "\n";

        mWriteFields = 1;
    } else {
        mWriteFields = 0;
    }
    mTimeStep += 1;
}

template<typename Field> void VtkOutput::write_field(int n_cells,
    const char* data_name, Field f[]) {
    if (mWriteFields) {
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
    if (mWriteFields) {
        std::ofstream file(mCurrentFile, std::ios_base::app);
        assert(file.is_open());

        file << "\nLINES " << n_connections << " " << 3*n_connections << "\n";
        for (int i = 0; i < n_connections; i++)
            file << "2 " << connections[i][0] << " " << connections[i][1] << "\n";
    }
}
