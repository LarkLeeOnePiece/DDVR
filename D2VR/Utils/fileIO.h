
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iomanip> // 注意需要包含这个头文件

// for 2D image
void read_sinogram(const std::string& filename, std::vector<double>& sino, int& n, int& n_angles);
void write_image(const std::string& filename, const std::vector<double>& image, int n);
void write_matrix(const std::string& filename, const double* data, int rows, int cols);

// for 3D image
void readFilteredSinograms(const std::string& filename, std::vector<double>& data,
    int num_slices, int num_detectors, int num_angles);


void save3DReconToBinary(const std::string& filename,const std::vector<double>& recon_host);

