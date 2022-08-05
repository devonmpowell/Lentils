#ifndef _NUFFT_H_
#define _NUFFT_H_

#include "gsl/gsl_sf_bessel.h"
#include "nufft.h"

// Options for gridding/degridding direction
//#define VIS_TO_GRID 0
//#define GRID_TO_VIS 1

#define FORWARD 0 
#define ADJOINT 1 

typedef struct {

	// grid sizes
	int nx_im, ny_im;
	int padx, pady;
	int nx_pad, ny_pad;
	int half_ny_pad;
	double pad_factor;

	// cell sizes
	double dx, dy;
	double cx, cy;
	double du, dv;

	// visibility size
	int nchannels;
	int nstokes;
	int nrows;
	double *uv;
	double *channels;

	// kernel-dependent parameters
	double wsup; // kernel support radius
	double kb_beta; // kaiser-Besser beta parameter 

} nufft_pars;


void init_nufft(nufft_pars *nufft, int nx, int ny, double xmin, double ymin, double xmax, double ymax, 
		double pad_factor, int kernel_support, int nrows, int nch, int nstokes, double *uv, double *channels);

void evaluate_apodization_correction(nufft_pars *nufft, double *out);

void zero_pad(nufft_pars *nufft, double *in, double *out, int direction);

double kb_g(double argk, double argq, double beta, double wsup);

double kb_w(double argx, double argy, double beta, double wsup);

double kb_optimal_beta(double padfac, double wsup);

void grid_cpu(nufft_pars *nufft, double *vis, double *grid, int direction);




#endif // _NUFFT_H_

