// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
//#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
// constants
extern const unsigned int window_width;
extern const unsigned int window_height;

extern const unsigned int mesh_width;
extern const unsigned int mesh_height;

extern float4 *h_vec;
extern float4 *d_vec;

extern float3 h_axis;
extern float3 *d_axis;
extern float  h_axis_radius;
extern GLfloat gkLightPos[4];
extern GLfloat gkLightPos2[4];

// vbo variables
extern GLuint vbo;
extern struct cudaGraphicsResource *cuda_vbo_resource;
extern void *d_vbo_buffer;

extern float g_fAnim;

// mouse controls
extern int mouse_old_x, mouse_old_y;
extern int mouse_buttons;
extern float rotate_x, rotate_y;
extern float translate_z;

extern StopWatchInterface *timer;

// Auto-Verification Code
extern int fpsCount;        // FPS count for averaging
extern int fpsLimit;        // FPS limit for sampling
extern int g_Index;
extern float avgFPS;
extern unsigned int frameCount;
extern unsigned int g_TotalErrors;
extern bool g_bQAReadback;

extern int *pArgc;
extern char **pArgv;

#define MAX(a,b) ((a > b) ? a : b)

extern const char *sSDKsample;

//ÉâÉCÉgíËêî
extern const GLfloat gkLightDiff[];
extern const GLfloat gkLightDiff2[];
//GLOBAL const GLfloat gkLightAmb[] ={ 0x01, 0x01, 0x01, 0x00};
//GLOBAL const GLfloat gkMaterial[] = { 1.0f, 1.0f, 1.0f, 0.2f}; 

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

void computeFPS();
