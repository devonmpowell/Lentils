
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <fftw3.h>
#include <omp.h>
#include "common.h"
#include "nufft.h"


double kb_g(double argk, double argq, double beta, double wsup) {
	if(!(argk >= -wsup && argk <= wsup && argq >= -wsup && argq <= wsup)) 
		return 0.0;
	double sqk = 1.0-(argk/wsup)*(argk/wsup);
	double sqq = 1.0-(argq/wsup)*(argq/wsup);
	if(sqk < 0.0) sqk = 0.0;
	if(sqq < 0.0) sqq = 0.0;
	double bk = beta*sqrt(sqk);
	double bq = beta*sqrt(sqq);
	return 1.0/(4.0*wsup*wsup)*gsl_sf_bessel_I0(bk)*gsl_sf_bessel_I0(bq);
}

double kb_w(double argx, double argy, double beta, double wsup) {

	double sqx = sqrt(beta*beta-(TWO_PI*wsup*argx)*(TWO_PI*wsup*argx));
	double sqy = sqrt(beta*beta-(TWO_PI*wsup*argy)*(TWO_PI*wsup*argy));
	double sinhcx, sinhcy;
	if(fabs(sqx) < 1.0e-6) sinhcx = 1.0;
	else sinhcx = sinh(sqx)/sqx;
	if(fabs(sqy) < 1.0e-6) sinhcy = 1.0;
	else sinhcy = sinh(sqy)/sqy;
	return sinhcx*sinhcy; 
}


double kb_optimal_beta(double padfac, double wsup) {
	return PI*sqrt(((2.0*wsup)/padfac)*((2.0*wsup)/padfac)*(padfac-0.5)*(padfac-0.5)-0.8);
}


void fft2d(nufft_pars *nufft, double *data_x, double *data_k, int direction) {


	// Execute FFT
	fftw_plan plan = NULL;
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads()/2);
	if(direction == FORWARD) {
		plan = fftw_plan_dft_r2c_2d(nufft->nx_pad, nufft->ny_pad, data_x, (fftw_complex*)data_k, FFTW_FORWARD|FFTW_ESTIMATE);
	}
	else if(direction == ADJOINT) {
		plan = fftw_plan_dft_c2r_2d(nufft->nx_pad, nufft->ny_pad, (fftw_complex*)data_k, data_x, FFTW_BACKWARD|FFTW_ESTIMATE);
	}
	if(plan) {
		fftw_execute(plan);
		fftw_destroy_plan(plan);
	}
	else {
		// TODO: errors
	}

	fftw_cleanup_threads();

}


void evaluate_apodization_correction(nufft_pars *nufft, double *out) {

	int i, j, idx;
	double x, y, val;
	for(i = 0; i < nufft->nx_im; ++i)
	for(j = 0; j < nufft->ny_im; ++j) {

		// c (row-major) ordering
		idx = nufft->ny_im*i + j;
		x = (i - 0.5*nufft->nx_im)/nufft->nx_pad; 
		y = (j - 0.5*nufft->ny_im)/nufft->ny_pad; 
		val = 1.0/kb_w(x, y, nufft->kb_beta, nufft->wsup);
		out[idx] = val;
	}
}

#if 1
void zero_pad(nufft_pars *nufft, double *in, double *out, int direction) {

	int iim, jim, idxim, idxpad;
	if(direction == FORWARD) {
		memset(out, 0, nufft->nx_pad*nufft->ny_pad*sizeof(double));
	}
	for(iim = 0; iim < nufft->nx_im; ++iim)
	for(jim = 0; jim < nufft->ny_im; ++jim) {
		// c (row-major) ordering
		idxim = nufft->ny_im*iim + jim;
		idxpad = nufft->ny_pad*(nufft->padx+iim) + (nufft->pady+jim);
		if(direction == FORWARD) {
			out[idxpad] = in[idxim];
		}
		else if(direction == ADJOINT) {
			out[idxim] = in[idxpad];
		}
	}
}
#endif

void convolution_matrix_csr(int im_nx, int im_ny, int k_nx, int k_ny, double *kernel, int *row_inds, int *cols, double *vals)
{

	int row, col, i, iim, jim, ik, jk, icc, jcc;
	int k_xmid = (k_nx+1)/2;
	int k_ymid = (k_ny+1)/2;
	for(iim = 0, i = 0, row = 0; iim < im_nx; ++iim)
	for(jim = 0; jim < im_ny; ++jim) {
		row_inds[row++] = i;
		for(ik = 0; ik < k_nx; ++ik)
		for(jk = 0; jk < k_ny; ++jk) {
			icc = iim + ik - k_xmid;
			jcc = jim + jk - k_ymid;
			if(icc < 0 || icc >= im_nx) continue;
			if(jcc < 0 || jcc >= im_ny) continue;
			col = im_ny*icc + jcc;
			vals[i] = kernel[k_ny*ik+jk]; 
			cols[i++] = col;
		}
	}
	row_inds[row] = i;
}



void init_nufft(nufft_pars *nufft, int nx, int ny, double xmin, double ymin, double xmax, double ymax, 
		double pad_factor, int kernel_support, int nrows, int nch, int nstokes, double *uv, double *channels) {
	

	// get padded dimensions
	nufft->nx_im = nx;
	nufft->ny_im = ny;
	nufft->padx = floor(((pad_factor-1.0)*nx+1)/2); // round up in dividing by 2 for odd-numbered grids or weird pad factors 
	nufft->pady = floor(((pad_factor-1.0)*ny+1)/2);
	nufft->nx_pad = nx+2*nufft->padx;
	nufft->ny_pad = nx+2*nufft->pady;
	nufft->half_ny_pad = nufft->ny_pad/2+1; 
	nufft->pad_factor = pad_factor;

	// cell sizes and center point
	// assumes input dimensions are in arcsec
	// converts to radians for internal use
	// TODO: check what the units should be
	nufft->dx = (xmax-xmin)/nx*ARCSEC_TO_RADIANS; 
	nufft->dy = (ymax-ymin)/ny*ARCSEC_TO_RADIANS; 
	nufft->cx = 0.5*(xmax+xmin)*ARCSEC_TO_RADIANS;
	nufft->cy = 0.5*(ymax+ymin)*ARCSEC_TO_RADIANS;
	nufft->du = 1.0/(nufft->dx*nufft->nx_pad); 
	nufft->dv = 1.0/(nufft->dy*nufft->ny_pad); 

	// set up uv info
	nufft->nchannels = nch;
	nufft->nstokes = nstokes;
	nufft->nrows = nrows;
	nufft->uv = uv;
	nufft->channels = channels;

	// set up kernel parameters
	nufft->wsup = kernel_support;
	nufft->kb_beta = kb_optimal_beta(nufft->pad_factor, nufft->wsup);

}




//#ifndef GPU
void grid_cpu(nufft_pars *nufft, double *vis, double *grid, int direction) {

	int i, flatidx, visidx, loopidx_u, loopidx_v, i_u, i_v, ngp_u, ngp_v;
	double u_tmp, v_tmp, cwgt_re, cwgt_im, kernel, pcenter;
	double du = nufft->du;
	double dv = nufft->dv;

	int ch_vis, ch_grid, st_vis, st_grid, stokes;

	for(ch_vis = 0, ch_grid = 0; ch_vis < nufft->nchannels; ++ch_vis) {
	//for(st_vis = 0, st_grid = 0; st_vis < nufft->nstokes; ++st_vis) {
	stokes = 0; // TODO

#pragma omp parallel for private(i,u_tmp,v_tmp,ngp_u,ngp_v,loopidx_u,loopidx_v,i_u,i_v,kernel,pcenter,cwgt_re,cwgt_im,flatidx,visidx) 
		for(i = 0; i < nufft->nrows; ++i) {

			// loop over the local uv grid with the gridding kernel
			// process nufft->wsup points on each side of the NGP
			u_tmp = nufft->uv[3*i+0]*nufft->channels[ch_vis];
			v_tmp = -nufft->uv[3*i+1]*nufft->channels[ch_vis];
			//double w_tmp = nufft->uv[3*i+2]*nufft->channels[ch_vis];
			ngp_u = round(u_tmp/nufft->du);
			ngp_v = round(v_tmp/nufft->dv);
			for(loopidx_u = ngp_u - nufft->wsup; loopidx_u <= ngp_u + nufft->wsup; ++loopidx_u)
			for(loopidx_v = ngp_v - nufft->wsup; loopidx_v <= ngp_v + nufft->wsup; ++loopidx_v) {

				// kernel and weights 
				// Also factor to center phases
				// Use a factor of 0.5 so that we don't double-count 
				// when forming cos() from exp(i*...) 
				i_u = loopidx_u;
				i_v = loopidx_v;
				kernel = kb_g((u_tmp/du-i_u), (v_tmp/dv-i_v), nufft->kb_beta, nufft->wsup);
				pcenter = 1-2*((i_u+i_v)&1);
				cwgt_re = 0.5*kernel*pcenter; 
				cwgt_im = 0.5*kernel*pcenter; 

				// Wrap. Beware of aliasing, the de-apodization kernel does not like it 
				while(i_u < 0) i_u += nufft->nx_pad; 
				while(i_v < 0) i_v += nufft->ny_pad; 
				while(i_u >= nufft->nx_pad) i_u -= nufft->nx_pad;
				while(i_v >= nufft->ny_pad) i_v -= nufft->ny_pad;

				// fold since we are using compressed c2r format
				if(i_v > nufft->half_ny_pad) {
					i_u = (nufft->nx_pad - i_u) % nufft->nx_pad;
					i_v = nufft->ny_pad - i_v; 
					cwgt_im *= -1;
				} 
				if(i_v == 0) {
					cwgt_re *= 2;
					cwgt_im *= 2;
				}

				// we have the weights, now grid or de-grid
				flatidx = nufft->half_ny_pad*i_u + i_v; // TODO: properly index with multiple channels/stokes
				visidx = nufft->nrows*nufft->nchannels*stokes+nufft->nrows*ch_vis+i;
				if(direction == FORWARD) {
#pragma omp atomic
					vis[2*visidx+0] += cwgt_re*grid[2*flatidx+0];
#pragma omp atomic
					vis[2*visidx+1] += cwgt_im*grid[2*flatidx+1];
				}
				else if(direction == ADJOINT) {
					
					// TODO: move this translation elsewhere
					double vis_re = vis[2*visidx+0];
					double vis_im = vis[2*visidx+1];
					double ccx = nufft->cx;
					double ccy = nufft->cy;
					double cosp = cos(TWO_PI*(u_tmp*ccx + v_tmp*ccy));
					double sinp = sin(TWO_PI*(u_tmp*ccx + v_tmp*ccy));
#pragma omp atomic
					grid[2*flatidx+0] += cwgt_re*(cosp*vis_re - sinp*vis_im);
#pragma omp atomic
					grid[2*flatidx+1] += cwgt_im*(sinp*vis_re + cosp*vis_im);
				}
			}
		}
	}
}

