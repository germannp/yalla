#include <cassert>
#include <fstream>


void write_positions(const char* file_name, int n_cells, float3 X[]) {
    std::ofstream file(file_name);
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

    file.close();
}
