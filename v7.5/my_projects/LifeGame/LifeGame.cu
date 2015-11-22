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
cudaProcess(unsigned int *g_odata, int imgw, int *dst, int *src, int WIDTH, int HEIGHT)
{
    extern __shared__ uchar4 sdata[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;

	if (x <= 0 || WIDTH <= x || y <= 0 || HEIGHT <= y)
		return;

	int s = src[(y - 1) + (x - 1) * HEIGHT] + src[(y - 1) + x * HEIGHT] + src[(y - 1) + (x + 1) * HEIGHT]
		+ src[y + (x - 1) * HEIGHT] + src[y + (x + 1) * HEIGHT]
		+ src[(y + 1) + (x - 1) * HEIGHT] + src[(y + 1) + x * HEIGHT] + src[(y + 1) + (x + 1) * HEIGHT];

	switch (s) {
	case 2:	// ˆÛŽ
		dst[y + x * HEIGHT] = src[y + x * HEIGHT];
		break;
	case 3:	// ’a¶
		dst[y + x * HEIGHT] = 1;
		break;
	default: // Ž€–Å
		dst[y + x * HEIGHT] = 0;
		break;
	}


    uchar4 c4 = (dst[y + x * HEIGHT] == 1) ? make_uchar4(255,255,255,0) : make_uchar4(0,0,0,0);
    g_odata[y*imgw+x] = rgbToInt(c4.z, c4.y, c4.x);
}

extern "C" void
launch_cudaProcess(dim3 grid, dim3 block, int sbytes,
                   unsigned int *g_odata,
                   int imgw, int *dst, int *src, int WIDTH, int HEIGHT)
{
	unsigned int mem_size = (WIDTH + 1) * (HEIGHT + 1) * sizeof(int);

	int*	d_dst;
	int*	d_src;

	cudaMalloc((void**)&d_dst, mem_size);
	cudaMalloc((void**)&d_src, mem_size);
	cudaMemcpy(d_dst, dst, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_src, src, mem_size, cudaMemcpyHostToDevice);

    cudaProcess<<< grid, block, sbytes >>>(g_odata, imgw, d_dst, d_src, WIDTH, HEIGHT);

	// copy results from device to host
	cudaMemcpy(dst, d_dst, mem_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(src, d_src, mem_size, cudaMemcpyDeviceToHost);

	// cleanup memory
	cudaFree(d_dst);
	cudaFree(d_src);
}
