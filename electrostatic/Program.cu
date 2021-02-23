#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// OpenGL headers
#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA headers
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Field.h"
int WIDTH = 512;
int HEIGHT = 550;
const int N = 100;

Field field = Field(N, WIDTH, HEIGHT);
GLuint pixelsPBO;
cudaGraphicsResource* pixelsPBOResource = NULL;


extern void ColorPixel_caller(cudaGraphicsResource* pixelsPBO, Field& f);
void cleanupAndExit();


void display()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();

	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelsPBO);
	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glutSwapBuffers();
}

void reshape(int w, int h)
{
	WIDTH = w;
	HEIGHT = h;
	field.resize(WIDTH, HEIGHT);

	// realokacja bufora PBO i ponowna jego rejestracja przez CUDA
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glDeleteBuffers(1, &pixelsPBO);
	glGenBuffers(1, &pixelsPBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelsPBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 3 * sizeof(GLubyte), NULL, GL_STREAM_DRAW);
	cudaError_t cudaStatus;

	if (pixelsPBOResource != NULL)
	{
		cudaStatus = cudaGraphicsUnregisterResource(pixelsPBOResource);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaGraphicsUnregisterResource error: %s\n", cudaGetErrorString(cudaStatus));
			cleanupAndExit();
		}
	}
	cudaStatus = cudaGraphicsGLRegisterBuffer(&pixelsPBOResource, pixelsPBO, cudaGraphicsRegisterFlagsWriteDiscard);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGraphicsGLRegisterBuffer error\n");
		cleanupAndExit();
	}

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, WIDTH, HEIGHT);

	gluOrtho2D(0, WIDTH, 0, HEIGHT);
	//gluOrtho2D(0, WIDTH, WIDTH, 0);
	glMatrixMode(GL_MODELVIEW);
	glutPostRedisplay();
}

int frameCount = 0;
const int text_len = 30;
char text[text_len];
void set_text()
{
	snprintf(text, text_len, "Electrostatic field, FPS: %d\n", frameCount);
	glutSetWindowTitle(text);
}

time_t T0 = time(NULL);
time_t now;
void timer(int)
{
	ColorPixel_caller(pixelsPBOResource, field);

	field.Move();

	glutPostRedisplay();

	frameCount++;
	now = time(NULL);
	if (now - T0 >= 1)
	{
		set_text();
		frameCount = 0;
		T0 = time(NULL);
	}
	
	glutTimerFunc(1000 / 60, timer, 0);
}

void gl_initialize(int* argc, char** argv)
{
	// utworzenie okna
	glutInit(argc, argv);
	glutInitWindowPosition(-1, -1);
	glutInitWindowSize(WIDTH, HEIGHT);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutCreateWindow("Electrostatic field");
	set_text();

	// handlers
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutTimerFunc(1000, timer, 0);

	// inicjalizacja GLEW
	if (GLEW_OK != glewInit()) { exit(EXIT_FAILURE); }
	while (GL_NO_ERROR != glGetError()); /* glewInit may cause some OpenGL errors -- flush the error state */

}

void cuda_initialize()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice error\n");
		cleanupAndExit();
	}
}

int main(int argc, char** argv)
{

	// zainicjowanie GLUTa i GLEW
	gl_initialize(&argc, argv);
	// zainicjowanie CUDA
	cuda_initialize();

	glutMainLoop();
	cleanupAndExit();
}

void cleanupAndExit()
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glDeleteBuffers(1, &pixelsPBO);

	exit(EXIT_SUCCESS);
}
