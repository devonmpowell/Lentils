#ifndef _COMMON_H_
#define _COMMON_H_


// math constants
#define ARCSEC_TO_RADIANS 4.8481368111e-6
#define RADIANS_TO_ARCSEC 206264.806 
#define PI 3.14159265359
#define TWO_PI 6.28318530718



typedef struct {

	int nx, ny;
	int nchannels;
	int nstokes;
	double xmin, xmax, ymin, ymax;
	double dx, dy;
	double *channels;
	char *mask;

} image_space;


typedef struct {

	// grid sizes
	int num_points, num_tris;
	int nchannels;
	int nstokes;
	double *channels;
	double *points;
	int *triangles;

} delaunay_space;


typedef struct {

	int nu, nv;
	int half_nv;
	int nchannels;
	int nstokes;
	double du, dv;
	double gcx, gcy;
	double *channels;

} fourier_space;


typedef struct {
	
	// uv space
	int nrows;
	int nchannels;
	int nstokes;
	double *uv;
	double *channels;

} visibility_space;


#endif // _COMMON_H_

