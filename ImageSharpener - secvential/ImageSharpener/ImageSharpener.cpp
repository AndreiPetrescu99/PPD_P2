
#include <iostream>
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

using std::chrono::high_resolution_clock;


int main()
{

//	double mask[3][3] = { 1 / 9.0, 1 / 9.0, 1 / 9.0,
//						 1 / 9.0, 1 / 9.0, 1 / 9.0,
//						 1 / 9.0, 1 / 9.0, 1 / 9.0 };

	double mask[3][3] = { 0.0, -1.0, 0.0,
						-1.0, 5.0, -1.0,
						 0.0, -1.0, 0.0 };

	double kernel[9];
	for (int ky = 0; ky <= 2; ky++) {
		for (int kx = 0; kx <= 2; kx++) {
			kernel[ky*3 + kx] = mask[ky][kx];
		}
	}

	int width, height, bpp;

	unsigned char* rgb_image = stbi_load("test2.png", &width, &height, &bpp, 0);
	unsigned char* image_out = (unsigned char*)malloc(width * height * bpp * sizeof(unsigned char));

	auto timestart = high_resolution_clock::now();

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < bpp; c++) {
				image_out[(y*width + x)*bpp + c] = 0.0f;
				double out = 0.0;
				for (int ky = -1; ky <= 1; ky++) {
					for (int kx = -1; kx <= 1; kx++) {
						out += rgb_image[((y + ky)*width + (x + kx))*bpp + c] * kernel[(ky + 1)*3 + (kx + 1)];
					}
				}
				if (out > 255.0) {
					out = 255.0;
				}
				if (out < 0.0) {
					out = 0.0;
				}
				image_out[(y*width + x)*bpp + c] = out;
			}
		}
	}

	auto timeend = high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(timeend - timestart).count();



	stbi_write_png("imageOut.png", width, height, 3, image_out, width*bpp);

	stbi_image_free(rgb_image);
    //std::cout << "Hello World!\n";
}