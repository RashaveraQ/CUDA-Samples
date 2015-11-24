/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Utilities and system includes

#include <helper_cuda.h>

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    return (int(b)<<16) | (int(g)<<8) | int(r);
}

__global__ void
cudaProcess(unsigned int *g_odata, int *dst, int *src, int WIDTH, int HEIGHT, int mouse_buttons, int mouse_x, int mouse_y)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;

	if (x <= 0 || WIDTH <= x || y <= 0 || HEIGHT <= y)
		return;

	int s;
	switch (mouse_buttons) {
	case 1:
		s = (x == mouse_x && y == mouse_y) ? 3 : 2;
		break;
	case 4:
		s = (x == mouse_x && y == mouse_y) ? 4 : 2;
		break;
	default:
		s = src[(y - 1) * WIDTH + (x - 1)] + src[(y - 1) * WIDTH + x] + src[(y - 1) * WIDTH + (x + 1)]
			+ src[y * WIDTH + (x - 1)] + src[y * WIDTH + (x + 1)]
			+ src[(y + 1) * WIDTH + (x - 1)] + src[(y + 1) * WIDTH + x] + src[(y + 1) * WIDTH + (x + 1)];
		break;
	}

	switch (s) {
	case 2:	// ˆÛŽ
		dst[y * WIDTH + x] = src[y * WIDTH + x];
		break;
	case 3:	// ’a¶
		dst[y * WIDTH + x] = 1;
		break;
	default: // Ž€–Å
		dst[y * WIDTH + x] = 0;
		break;
	}

    uchar4 c4 = (dst[y * WIDTH + x] == 1) ? make_uchar4(255,255,255,0) : make_uchar4(0,0,0,0);
	g_odata[y * WIDTH + x] = rgbToInt(c4.z, c4.y, c4.x);
}

extern "C" void
launch_cudaProcess(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int *d_dst, int *d_src, int WIDTH, int HEIGHT, int mouse_buttons, int mouse_x, int mouse_y)
{
    cudaProcess<<< grid, block, sbytes >>>(g_odata, d_dst, d_src, WIDTH, HEIGHT, mouse_buttons, mouse_x, mouse_y);
}

__global__ void
cudaProcess_setPixel(unsigned int *g_odata, int imgw, int x, int y, bool set)
{
	uchar4 c4 = set ? make_uchar4(255, 255, 0, 0) : make_uchar4(20, 20, 20, 0);
	g_odata[y*imgw+x] = rgbToInt(c4.z, c4.y, c4.x);
}

extern "C" void
launch_cudaProcess_setPixel(unsigned int *g_odata, int imgw, int x, int y, bool set)
{
	cudaProcess_setPixel <<<1, 0 >>>(g_odata, imgw, x, y, set);
}