
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "math.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>
#include <ctime>
#include <iostream>
using namespace std;

cudaError_t addWithCuda(int width, int height, int bpp, double* kernel, unsigned char* rgb_image, unsigned char* image_out);

using std::chrono::high_resolution_clock;

__global__ void addKernel(double* kernel, unsigned char* rgb_image, unsigned char* image_out, int* width, int* height)
{

    int x = threadIdx.x;
	int y = blockIdx.x;
	//int width = blockDim.x;
	//printf("%d %d \n", x, y);
	//printf("%d,%d \n", *height, *width);

	if (y < *height && x < *width) {
		//printf("%d %d \n", x, y);
		for (int c = 0; c < 3; c++) {
			image_out[(y* *width + x) * 3 + c] = 0.0f;
			double out = 0.0;
			for (int ky = -1; ky <= 1; ky++) {
				for (int kx = -1; kx <= 1; kx++) {
					out += rgb_image[((y + ky)* *width + (x + kx)) * 3 + c] * kernel[(ky + 1) * 3 + (kx + 1)];
				}
			}
			if (out > 255.0) {
				out = 255.0;
			}
			if (out < 0.0) {
				out = 0.0;
			}
			image_out[(y* *width + x) * 3 + c] = out;
		}
	}
    
}

int main()
{
	double mask[3][3] = { 0.0, -1.0, 0.0,
					-1.0, 5.0, -1.0,
					 0.0, -1.0, 0.0 };

	double* kernel = (double*)malloc(9*sizeof(double));
	for (int ky = 0; ky <= 2; ky++) {
		for (int kx = 0; kx <= 2; kx++) {
			kernel[ky*3 + kx] = mask[ky][kx];
		}
	}

	int width, height, bpp;
	unsigned char* rgb_image = stbi_load("test2.png", &width, &height, &bpp, 0);
	unsigned char* image_out = (unsigned char*)malloc(width * height * bpp * sizeof(unsigned char));

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, int width*height*bpp);

	const size_t malloc_limit = size_t(2048) * size_t(2048) * size_t(2048);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, malloc_limit);


	auto timestart = high_resolution_clock::now();
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(width, height, bpp, kernel, rgb_image, image_out);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	
	auto timeend = high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(timeend - timestart).count();

	stbi_write_png("imageOut.png", width, height, 3, image_out, width*bpp);

	stbi_image_free(rgb_image);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int width, int height, int bpp, double* kernel, unsigned char* rgb_image, unsigned char* image_out)
{
	double* dev_kernel = 0;
	unsigned char* dev_rgb_image = 0;
	unsigned char* dev_image_out = 0;
	int* dev_width = 0;
	int* dev_height = 0;

    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


    // Allocate GPU buffers for three vectors (two input, one output)    .

	cudaStatus = cudaMalloc((void**)&dev_width, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_width, &width, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_height, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_height, &height, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

    cudaStatus = cudaMalloc((void**)&dev_kernel, 9 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_kernel, kernel, 9 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

    cudaStatus = cudaMalloc((void**)&dev_rgb_image, width * height * bpp * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_image_out, width * height * bpp * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_rgb_image, rgb_image, width * height * bpp * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	dim3 blocksPerGrid(height, 1, 1);
	dim3 threadsPerBlock(width, 1, 1);

    addKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_kernel, dev_rgb_image, dev_image_out, dev_width, dev_height);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(image_out, dev_image_out, height * width * bpp * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
   }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
