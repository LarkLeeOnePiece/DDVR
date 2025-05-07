#include "fileIO.h"

//------------------
// for 2D image
void read_sinogram(const std::string& filename, std::vector<double>& sino, int& n, int& n_angles) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    in.read((char*)&n, sizeof(int));
    in.read((char*)&n_angles, sizeof(int));

    std::cout << "[Debug] Read n = " << n << ", n_angles = " << n_angles << std::endl;

    sino.resize(n * n_angles);
    in.read((char*)sino.data(), n * n_angles * sizeof(double));
    in.close();
}

void write_image(const std::string& filename, const std::vector<double>& image, int n) {
    std::ofstream out(filename, std::ios::binary);
    out.write((char*)&n, sizeof(int));
    out.write((char*)image.data(), n * n * sizeof(double));
    out.close();
}
void write_matrix(const std::string& filename, const double* data, int rows, int cols) {
    std::ofstream out(filename, std::ios::binary);
    out.write((char*)&rows, sizeof(int));
    out.write((char*)&cols, sizeof(int));
    out.write((char*)data, sizeof(double) * rows * cols);
    out.close();
}
//------------------

//  ---------------------
// Here we have 3D data as input
void readFilteredSinograms(const std::string& filename, std::vector<double>& data,
    int num_slices, int num_detectors, int num_angles)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    size_t total_size = static_cast<size_t>(num_slices) * num_detectors * num_angles;
    data.resize(total_size);

    file.read(reinterpret_cast<char*>(data.data()), total_size * sizeof(double));
    file.close();

    std::cout << "Read " << total_size << " double values from " << filename << std::endl;
}

void save3DReconToBinary(const std::string& filename,
    const std::vector<double>& recon_host)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    ofs.write(reinterpret_cast<const char*>(recon_host.data()),
        recon_host.size() * sizeof(double));

    ofs.close();
    std::cout << "Saved binary file: " << filename << " (" << recon_host.size() << " doubles)\n";
}