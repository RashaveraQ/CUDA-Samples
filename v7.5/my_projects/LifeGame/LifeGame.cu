﻿/*
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
	return (int(b) << 16) | (int(g) << 8) | int(r);
}

__device__ int idx(int x, int y, int width, int height)
{
	x = (x < 0) ? (width - 1) : (width <= x) ? 0 : x;
	y = (y < 0) ? (height - 1) : (height <= y) ? 0 : y;
	return y * width + x;
}

__global__ void
cudaProcess(unsigned int *g_odata, int *dst, int *src, int width, int height, int mouse_buttons, int mouse_x, int mouse_y, bool is_running)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	if (x < 0 || width <= x || y < 0 || height <= y)
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
		s = (is_running) ?
			src[idx(x - 1, y - 1, width, height)] + src[idx(x, y - 1, width, height)] + src[idx(x + 1, y - 1, width, height)]
			+ src[idx(x - 1, y, width, height)] + src[idx(x + 1, y, width, height)]
			+ src[idx(x - 1, y + 1, width, height)] + src[idx(x, y + 1, width, height)] + src[idx(x + 1, y + 1, width, height)]
			: 2;
		break;
	}

	int c = idx(x, y, width, height);
	switch (s) {
	case 2:	// 維持
		dst[c] = src[c];
		break;
	case 3:	// 誕生
		dst[c] = 1;
		break;
	default: // 死滅
		dst[c] = 0;
		break;
	}

	uchar4 c4 = (dst[c] == 1) ? make_uchar4(128, 128, 128, 0) : make_uchar4(0, 0, 0, 0);
	g_odata[c] = rgbToInt(c4.z, c4.y, c4.x);
}

extern "C" void
launch_cudaProcess(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int *d_dst, int *d_src, int WIDTH, int HEIGHT, int mouse_buttons, int mouse_x, int mouse_y, bool is_running)
{
	cudaProcess<<< grid, block, sbytes >>>(g_odata, d_dst, d_src, WIDTH, HEIGHT, mouse_buttons, mouse_x, mouse_y, is_running);
}

