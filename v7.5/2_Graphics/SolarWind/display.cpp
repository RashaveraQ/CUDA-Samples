#include "SolarWind.h"

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);

	//簡易ライトセット
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0, GL_POSITION, gkLightPos);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, gkLightDiff);
//	glLightfv(GL_LIGHT0, GL_AMBIENT, gkLightAmb);
	glEnable(GL_LIGHT1);
	glLightfv(GL_LIGHT1, GL_POSITION, gkLightPos2);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, gkLightDiff2);

	//Zバッファ有効
	glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// Earth
//	glMaterialfv(GL_FRONT, GL_DIFFUSE, gkMaterial);
	glutSolidSphere(50.0 * h_axis_radius, 20, 20);
	glDisable(GL_LIGHTING);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 1.0, 1.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}
