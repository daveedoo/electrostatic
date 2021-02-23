#pragma once
#include <vector_types.h>

class Field
{
public:
	ushort2* d_positions;
	float2* d_velocities;
	bool* d_sign;
	float2* d_fieldForce;

	int N;
	int width;
	int height;

	Field(int n, int maxX, int maxY);
	~Field();
	void Move();
	void resize(int newWidth, int newHeight);
};
