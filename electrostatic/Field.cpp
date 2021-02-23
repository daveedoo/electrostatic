#include "Field.h"
#include <stdlib.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <GL/freeglut.h>

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

void moveParticle_caller(Field& f);

void setSign(bool* arr, int i, bool val)
{
	int ind = 128 * (i / 128) + ((4 * (i % 128)) % 128) + ((i / 32) % 4);
	arr[ind] = val;
}

bool getSign(bool* arr, int i)
{
	int ind = 128 * (i / 128) + ((4 * (i % 128)) % 128) + ((i / 32) % 4);
	return arr[ind];
}


Field::Field(int n, int width, int height)
{
	N = n;
	this->width = width;
	this->height = height;

	bool* sign = (bool*)malloc(N * sizeof(bool));

	ushort2* pos = (ushort2*)malloc(N*sizeof(ushort2));
	float2* vel = (float2*)malloc(N * sizeof(float2));

	srand(time(NULL));
	for (int i = 0; i < n; i++)
	{
		pos[i] = make_ushort2(rand() % (width),
							rand() % (height));
		//setSign(sign, i, (rand() % 2 == 0) ? true : false);
		sign[i] = (rand() % 2 == 0) ? true : false;

		float angle = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (2 * M_PI)));
		vel[i] = make_float2(3.f*static_cast<float>(cos(angle)),
							3.f*static_cast<float>(sin(angle)));
	}

	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(&d_positions, N * sizeof(ushort2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
	}
	cudaStatus = cudaMemcpy(d_positions, pos, N * sizeof(ushort2), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed\n");
	}

	cudaStatus = cudaMalloc(&d_velocities, N * sizeof(float2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
	}
	cudaStatus = cudaMemcpy(d_velocities, vel, N * sizeof(float2), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed\n");
	}

	cudaStatus = cudaMalloc(&d_sign, N * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
	}
	cudaStatus = cudaMemcpy(d_sign, sign, N * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed\n");
	}

	cudaStatus = cudaMalloc(&d_fieldForce, width*height * sizeof(float2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
	}

	free(pos);
	free(vel);
	free(sign);
}

void Field::Move()
{
	moveParticle_caller(*this);
}

void Field::resize(int newWidth, int newHeight)
{
	width = newWidth;
	height = newHeight;

	cudaError_t cudaStatus;
	cudaStatus = cudaFree(d_fieldForce);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree -- d_fieldForce error\n");
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaMalloc(&d_fieldForce, width*height * sizeof(float2));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc -- d_fieldForce error");
		exit(EXIT_FAILURE);
	}
}

Field::~Field()
{
	cudaFree(d_fieldForce);
	cudaFree(d_sign);
	cudaFree(d_velocities);
	cudaFree(d_positions);
}
