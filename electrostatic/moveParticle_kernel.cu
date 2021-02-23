#include <stdio.h>
#include <cuda_runtime.h>

#include "Field.h"
//#define TIMER

__global__ void moveParticle_kernel(int N, ushort2* position, bool* sign, float2* velocity, float2* field, int width, int height)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		ushort2 pos = position[i];
		float2 vel = velocity[i];

		// aktualizacja predkosci
		int pixel_idx = pos.y * width + pos.x;
		float2 f = field[pixel_idx];
		if (sign[i])
		{
			vel.x += f.x;
			vel.y += f.y;
		}
		else
		{
			vel.x -= f.x;
			vel.y -= f.y;
		}

		// aktualizacja polozenia
		pos.x += vel.x;
		pos.y += vel.y;

		// sprawdzenie, czy czasteczka wyszla poza ramke
		if (pos.x > width - 2)
		{
			pos.x = width - 1;
			vel.x = -vel.x;
		}
		else if (pos.x < 1)
		{
			pos.x = 0;
			vel.x = -vel.x;
		}

		if (pos.y > height - 2)
		{
			pos.y = height - 1;
			vel.y = -vel.y;
		}
		else if (pos.y < 1)
		{
			pos.y = 0;
			vel.y = -vel.y;
		}

		position[i] = pos;
		velocity[i] = vel;
	}
}

void moveParticle_caller(Field& f)
{
	int threadsPerBlock = 64;
	int blocks = f.N / threadsPerBlock;
	if (f.N % threadsPerBlock != 0)
		blocks++;

#ifdef TIMER
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
#endif
	moveParticle_kernel<<<blocks, threadsPerBlock>>>(f.N, f.d_positions, f.d_sign, f.d_velocities, f.d_fieldForce, f.width, f.height);
#ifdef TIMER
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	printf("%f\n", time*1000);
#endif

	cudaError_t cudaStatus;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveParticle_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching moveParticle_kernel!\n", cudaStatus);
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
	}
}