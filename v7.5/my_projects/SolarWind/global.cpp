#include "SolarWind.h"

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 1024;
const unsigned int window_height = 1024;

const unsigned int mesh_width    = 256;
const unsigned int mesh_height   = 256;

float4 *h_vec;
float4 *d_vec;

float3 h_axis;
float3 *d_axis;
float  h_axis_radius;
GLfloat gkLightPos[4];
GLfloat gkLightPos2[4];

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 30.0, rotate_y = 15.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

const char *sSDKsample = "simpleGL (VBO)";

//ÉâÉCÉgíËêî
const GLfloat gkLightDiff[] ={ 0x0F, 0x00, 0x00, 0x00};
const GLfloat gkLightDiff2[] ={ 0x00, 0x00, 0x0F, 0x00};
//const GLfloat gkLightAmb[] ={ 0x01, 0x01, 0x01, 0x00};
//const GLfloat gkMaterial[] = { 1.0f, 1.0f, 1.0f, 0.2f}; 
