#pragma once
#include <stdio.h>
#include <math.h>

#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Field.h"
#define BLOCK_SIZE 256
//#define TIMER

__device__ inline float distanceSqrd(short x1, short y1, short x2, short y2)
{
	return (float)((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
}

// funkcje używane do ewentualnego lepszego dostępu do pamiędzi dzielonej
//__device__ inline bool getSign(bool* arr, int i)
//{
//	//return arr[128 * (i / 128) + ((4 * (i % 128)) % 128) + ((i / 32) % 4)];
//	return arr[4*i];
//}
//
//__device__ inline void setSign(bool* arr, int i, bool val)
//{
//	//arr[128 * (i / 128) + ((4 * (i % 128)) % 128) + ((i / 32) % 4)] = val;
//	arr[4*i] = val;
//}

__global__ void ColorPixel_kernel(int N, ushort2* d_positions, bool* d_sign, int k, GLubyte* d_pixels, float2* d_field, int width, int height)
{
	short x = blockIdx.x * blockDim.x + threadIdx.x;
	short y = blockIdx.y * blockDim.y + threadIdx.y;

	float Potential = 0;						// agregat na potencjał w danym pixelu
	float2 F = make_float2(0.f, 0.f);			// agregat na wektor natężenia w danym pixelu

	short maxP = N / BLOCK_SIZE;	// licznik przebiegów "największego" fora
	if (N % BLOCK_SIZE != 0)
		maxP++;
	for (short p = 0; p < maxP; p++)
	{
		__shared__ ushort2 position[BLOCK_SIZE];
		__shared__ bool sign[BLOCK_SIZE];
		//__shared__ bool sign[4*BLOCK_SIZE];

		short th_ind = threadIdx.y * blockDim.x + threadIdx.x;	// index wątku wew. bloku
		short p_ind = p*BLOCK_SIZE + th_ind;					// index kopiowanej cząstki
		
		// kopiowanie do shared memory
		if (p_ind < N)
		{
			position[th_ind] = d_positions[p_ind];
			sign[th_ind] = d_sign[p_ind];
			//setSign(sign, th_ind, d_sign[p_ind]);
		}
		__syncthreads();

		if (x < width && y < height)
		{
			short I;
			if (p == maxP - 1 && N < maxP*BLOCK_SIZE)
				I = N - (p*BLOCK_SIZE);
			else
				I = BLOCK_SIZE;

			float flen, distSqrd, dist;

			// właściwe obliczenia dla cząstki pod indeksem i
			for (int i = 0; i < I; i++)
			{
				if (x == position[i].x && y == position[i].y)
					continue;

				distSqrd = distanceSqrd(x, y, position[i].x, position[i].y);
				dist = sqrt(distSqrd);
				flen = 1.f / distSqrd;

				//if (getSign(sign, i))
				if (sign[i])
				{
					Potential += 1.f / dist;					// przyczynek do potencjału pola w punkcie.					Potencjał jest używany do wizualizacji.
					F.x -= (flen / dist) * (position[i].x - x);	// przyczynek do natężenia pola w punkcie, składowa x.		Wektor natężenia jest używany do przemieszczania się cząstek.
					F.y -= (flen / dist) * (position[i].y - y);	// przyczynek do natężenia pola w punkcie, składowa y

				}
				else
				{
					Potential -= 1.f / dist;
					F.x += (flen / dist) * (position[i].x - x);
					F.y += (flen / dist) * (position[i].y - y);
				}
			}
		}
		__syncthreads();
	}

	// końcowe zapisywanie danych
	if (x < width && y < height)
	{
		Potential = Potential * k;

		int mult = 10;
		F.x *= mult;
		F.y *= mult;

		int pixel_idx = y * width + x;
		d_field[pixel_idx] = F;

		if (Potential > 0)
		{
			((uchar3*)d_pixels)[pixel_idx] = make_uchar3(255, 255 - Potential, 255 - Potential);	// R G B
		}
		else
		{
			((uchar3*)d_pixels)[pixel_idx] = make_uchar3(255 + Potential, 255 + Potential, 255);	// R G B
		}
	}
}

void ColorPixel_caller(cudaGraphicsResource* pixelsPBO, Field& f)
{
	// mnoznik potencjału pola
	int k = 1000;

	int s = 16, t = 16;
	dim3 threadsPerBlock(s, t);
	dim3 blocks(f.width / s, f.height / t);
	if (f.width % s != 0)
		blocks.x += 1;
	if (f.height % t != 0)
		blocks.y += 1;

	GLubyte* d_pixels;
	size_t num_bytes;
	cudaGraphicsMapResources(1, &pixelsPBO, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &num_bytes, pixelsPBO);
#ifdef TIMER
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
#endif

	ColorPixel_kernel<<<blocks, threadsPerBlock>>>(f.N, f.d_positions, f.d_sign, k, d_pixels, f.d_fieldForce, f.width, f.height);

#ifdef TIMER
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("%f\n", time);
#endif

	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ColorPixel_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching ColorPixel_kernel!\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
		exit(EXIT_FAILURE);
	}

	cudaGraphicsUnmapResources(1, &pixelsPBO, 0);
}
