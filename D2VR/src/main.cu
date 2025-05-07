#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iomanip> // 注意需要包含这个头文件

// for our case
#include "../Utils/fileIO.h"
#include "globalParas.h"
#include <iostream>
#include <chrono>


#define PI 3.14159265358979323846f


__global__ void backproject_kernel(
    const double* sino,
    const double* angles,
    double* recon,
    int n,
    int n_angles
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= n || y >= n) return;

    double cx = n / 2.0f;
    double cy = n / 2.0f;
    double acc = 0.0f;

    for (int i = 0; i < n_angles; ++i) {
        double theta = angles[i];//Be careful about the input, make sure it is radians
        double s = (x - cx) * cos(theta) + (y - cy) * sin(theta) + cx;
        //s = fminf(fmaxf(s, 0.0f), n - 1.0f); // 裁剪到 [0, n-1]

        int si = (int)floorf(s);
        double ds = s - si;

        if (si >= 0 && si <= n-1) {

            if (si >= 0 && si < n - 1) {
                float val = sino[i + si * n_angles] * (1.0f - ds) + sino[i + (si + 1) * n_angles] * ds;
                acc += val;
            }
            else if (si == n - 1) {
                acc += sino[i + si * n_angles]; // 使用边界值，不插值
            }


            ////double val = sino[si + i * n] * (1.0f - ds) + sino[si + 1 + i * n] * ds;
            //int index1 = si * n_angles + i;
            //int index2 = (si + 1) * n_angles + i;
            //double val = sino[index1] * (1.0f - ds) + sino[index2] * ds;

            ////double val = sino[i + si * n_angles] * (1.0f - ds) + sino[i + (si+1) * n_angles] * ds;
            //acc += val;
        }
        else if (si < 0) {
            acc += sino[i + 0 * n_angles]; // 使用边界值，不插值
        }
        else if (si > n-1) {
            acc += sino[i + n-1 * n_angles]; // 使用边界值，不插值
        }
    }

    recon[y * n + x] = acc / n_angles*PI;
}

int two_DFBP() {
    int n, n_angles;
    std::vector<double> sino_host;
    read_sinogram("../data/filtered_sinogram.bin", sino_host, n, n_angles);
    std::cout << "Read Sinogram" << std::endl;
    std::cout << std::fixed << std::setprecision(8);  // 保留6位小数
    std::cout << "Sinogram first line:" << sino_host[0] << ',' << sino_host[1] << sino_host[2] << std::endl;
    std::cout << "Sinogram second line:" << sino_host[0 + n_angles] << ',' << sino_host[1 + n_angles] << sino_host[2 + n_angles] << std::endl;
    //ram_lak_filter(sino_host, n, n_angles);
    std::cout << "Filtered with Ram-Lak" << std::endl;
    write_matrix("filtered_sinogram.bin", sino_host.data(), n, n_angles);
    double* sino_gpu;
    double* angles_gpu;
    double* recon_gpu;
    std::vector<double> angles_host(n_angles);
    for (int i = 0; i < n_angles; ++i)
        angles_host[i] = i * 180.0f / n_angles;

    cudaMalloc(&sino_gpu, sizeof(double) * n * n_angles);
    cudaMalloc(&angles_gpu, sizeof(double) * n_angles);
    cudaMalloc(&recon_gpu, sizeof(double) * n * n);

    cudaMemcpy(sino_gpu, sino_host.data(), sizeof(double) * n * n_angles, cudaMemcpyHostToDevice);
    cudaMemcpy(angles_gpu, angles_host.data(), sizeof(double) * n_angles, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    backproject_kernel << <blocks, threads >> > (sino_gpu, angles_gpu, recon_gpu, n, n_angles);
    cudaDeviceSynchronize();

    std::vector<double> recon_host(n * n);
    cudaMemcpy(recon_host.data(), recon_gpu, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

    write_image("reconstruction.bin", recon_host, n);

    cudaFree(sino_gpu);
    cudaFree(angles_gpu);
    cudaFree(recon_gpu);

    std::cout << "Reconstruction complete.\n";
    return 0;
}

int FBP_3D(const std::string& filename) {

    readFilteredSinograms(filename, sinos_host,num_slices, num_detectors, num_angles);
    write_matrix("3d_cuda_filtered_proj_0.bin", sinos_host.data(), num_detectors, num_angles);//checked, for each sinos, it is in the shape(num_dectors,num_angles)
    //reconstruct slice by slice
    double* sinos_gpu;
    double* angles_gpu;
    double* recon_gpu;
    cudaMalloc(&sinos_gpu, sizeof(double) * num_detectors * num_angles * num_slices);
    cudaMalloc(&angles_gpu, sizeof(double) * num_angles);
    cudaMalloc(&recon_gpu, sizeof(double) * num_detectors * num_detectors * num_slices);
    std::vector<double> recon_host(num_detectors * num_detectors * num_slices);


    std::cout << "angles"<<"("<< num_angles<<"): ";
    for (int i = 0; i < num_angles; i++){
        angles_host[i] = double(i) / double(num_angles-1)*PI - PI / 2;
        std::cout << " "<<angles_host[i];
    }
    std::cout << std::endl;
    
    //initialize data
    cudaMemcpy(sinos_gpu, sinos_host.data(), sizeof(double) * num_detectors * num_angles * num_slices, cudaMemcpyHostToDevice);
    cudaMemcpy(angles_gpu, angles_host.data(), sizeof(double) * num_angles, cudaMemcpyHostToDevice);
    
    dim3 threads(16, 16);
    dim3 blocks((num_detectors + 15) / 16, (num_detectors + 15) / 16);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_slices; i++) {
        int slice_idx = i;
        double* sino_slice_gpu = sinos_gpu + slice_idx * num_detectors * num_angles;
        double* recon_slice_gpu = recon_gpu + slice_idx * num_detectors * num_detectors;
        backproject_kernel << <blocks, threads >> > (sino_slice_gpu, angles_gpu, recon_slice_gpu, num_detectors, num_angles);
        cudaDeviceSynchronize();

        
        cudaMemcpy(recon_host.data()+slice_idx * num_detectors * num_detectors, recon_gpu, sizeof(double) * num_detectors * num_detectors, cudaMemcpyDeviceToHost);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "run time: " << duration.count() << " s" << std::endl;
    save3DReconToBinary("3Dreconstruction.bin", recon_host);
    return 0;
}


int num_slices=60; //projs height
int num_detectors=290; //projs width
int num_angles=181; //projs angles
std::vector<double> sinos_host;
std::vector<double> angles_host(num_angles);
int main() {
    FBP_3D("../data/3D_filtered_projections.bin");
}
