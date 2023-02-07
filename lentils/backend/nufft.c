
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "gsl/gsl_sf_bessel.h"
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


void zero_pad(image_space imspace, double *image, image_space padspace, double *padded, int direction) {

	int iim, jim, idxim, idxpad;
	int padx = (padspace.nx-imspace.nx)/2;
	int pady = (padspace.ny-imspace.ny)/2;
	if(direction == FORWARD) {
		memset(padded, 0, padspace.nx*padspace.ny*sizeof(double));
	}
	for(iim = 0; iim < imspace.nx; ++iim)
	for(jim = 0; jim < imspace.ny; ++jim) {
		idxim = imspace.ny*iim + jim;
		idxpad = padspace.ny*(padx+iim) + (pady+jim);
		if(direction == FORWARD) {
			padded[idxpad] = image[idxim];
		}
		else if(direction == ADJOINT) {
			image[idxim] = padded[idxpad];
		}
	}
}


#if 0
void zpad_matrix_csr(image_space imspace, image_space padspace, int *row_inds, int *cols, double *vals) {

	int iim, jim, idxim, idxpad;
	int padx = (padspace.nx-imspace.nx)/2;
	int pady = (padspace.ny-imspace.ny)/2;


	//for(iim = 0; iim < imspace.nx; ++iim)
	//for(jim = 0; jim < imspace.ny; ++jim) {
		//idxim = imspace.ny*iim + jim;
		//idxpad = padspace.ny*(padx+iim) + (pady+jim);
		//if(direction == FORWARD) {
			//padded[idxpad] = image[idxim];
		//}
		//else if(direction == ADJOINT) {
			//image[idxim] = padded[idxpad];
		//}
	//}


}
#endif




void dft_matrix_csr(visibility_space visspace, image_space imspace, int *row_inds, int *cols, double *vals)
{

	int row, col, i, vis, iim, jim, odd;
	double x, y, u, v, arg, val;

	int ch_vis = 0;
	for(row = 0, i = 0; row < 2*visspace.nrows; ++row)  {

		vis = row/2;
		odd = row%2;
		row_inds[row] = i;

		u = visspace.uv[3*vis+0]*visspace.channels[ch_vis];
		v = -visspace.uv[3*vis+1]*visspace.channels[ch_vis];
		
		for(iim = 0, col = 0; iim < imspace.nx; ++iim)
		for(jim = 0; jim < imspace.ny; ++jim, ++col) {

			if(!imspace.mask[col]) continue;

			x = iim*imspace.dx + imspace.xmin;
			y = jim*imspace.dy + imspace.ymin;
			arg = -2.0*M_PI*ARCSEC_TO_RADIANS*(u*x+v*y);
			if(odd) val = sin(arg);
			else val = cos(arg);

			vals[i] = val; 
			cols[i] = col; 
			i++;
		}
	}
	row_inds[row] = i;
}

void grad_matrix_findiff_csr(image_space imspace, int *row_inds, int *cols, double *vals) {

	int row, col, i, iim, jim;
	double val;

	row = 0;
	i = 0;

	// x-component of gradient
	for(iim = 0; iim < imspace.nx; ++iim)
	for(jim = 0; jim < imspace.ny; ++jim, ++row) {
		row_inds[row] = i;
		if(iim < imspace.nx-1) {
			vals[i] = -1.0/imspace.dx; 
			cols[i++] = imspace.ny*(iim+0) + jim; 
			vals[i] = 1.0/imspace.dx; 
			cols[i++] = imspace.ny*(iim+1) + jim; 
		}
	}
	// y-component of gradient
	for(iim = 0; iim < imspace.nx; ++iim)
	for(jim = 0; jim < imspace.ny; ++jim, ++row) {
		row_inds[row] = i;
		if(jim < imspace.ny-1) {
			vals[i] = -1.0/imspace.dy; 
			cols[i++] = imspace.ny*iim + (jim+0); 
			vals[i] = 1.0/imspace.dy; 
			cols[i++] = imspace.ny*iim + (jim+1); 
		}
	}
	row_inds[row] = i;
}


void curv_matrix_findiff_csr(image_space imspace, int *row_inds, int *cols, double *vals) {

	int row, col, i, iim, jim;
	double val, invh2;

	// 5-point Laplacian stencil
	row = 0;
	i = 0;
	invh2 = 1.0/(imspace.dx*imspace.dy);
	for(iim = 0; iim < imspace.nx; ++iim)
	for(jim = 0; jim < imspace.ny; ++jim, ++row) {
		row_inds[row] = i;
		if(iim > 0) {
			vals[i] = invh2; 
			cols[i++] = imspace.ny*(iim-1) + jim; 
		}
		if(jim > 0) {
			vals[i] = invh2; 
			cols[i++] = imspace.ny*iim + (jim-1); 
		}
		vals[i] = -4.0*invh2; 
		cols[i++] = imspace.ny*iim + jim; 
		if(jim < imspace.ny-1) {
			vals[i] = invh2; 
			cols[i++] = imspace.ny*iim + (jim+1); 
		}
		if(iim < imspace.nx-1) {
			vals[i] = invh2; 
			cols[i++] = imspace.ny*(iim+1) + jim; 
		}
	}
	row_inds[row] = i;
}



void convolution_matrix_csr(image_space imspace, int k_nx, int k_ny, double *kernel, 
		int *row_inds, int *cols, double *vals) {

	int row, col, i, iim, jim, ik, jk, icc, jcc;
	int k_xmid = k_nx/2;
	int k_ymid = k_ny/2;
	for(iim = 0, i = 0, row = 0; iim < imspace.nx; ++iim)
	for(jim = 0; jim < imspace.ny; ++jim, ++row) {
		row_inds[row] = i;
		for(ik = 0; ik < k_nx; ++ik)
		for(jk = 0; jk < k_ny; ++jk) {
			// flip the kernel so that it is not transposed
			icc = iim + k_xmid - ik;
			jcc = jim + k_ymid - jk;
			col = imspace.ny*icc + jcc;
			if(icc < 0 || icc >= imspace.nx) continue;
			if(jcc < 0 || jcc >= imspace.ny) continue;
			if(!imspace.mask[col]) continue;
			vals[i] = kernel[k_ny*ik+jk]; 
			cols[i] = col;
			i++;
		}
	}
	row_inds[row] = i;
}





//#ifndef GPU
void grid_cpu(visibility_space visspace, double *vis, fourier_space gspace, double *grid, 
		int wsup, double kb_beta, int direction) {

	int i, flatidx, visidx, loopidx_u, loopidx_v, i_u, i_v, ngp_u, ngp_v;
	int ch_vis, ch_grid, st_vis, st_grid, stokes;
	double u_tmp, v_tmp, cwgt_re, cwgt_im, kernel, pcenter;
	double tmp_re, tmp_im, cosp, sinp;

	for(ch_vis = 0, ch_grid = 0; ch_vis < visspace.nchannels; ++ch_vis) {
	//for(st_vis = 0, st_grid = 0; st_vis < nufft->nstokes; ++st_vis) { // TODO
		stokes = 0;

#pragma omp parallel for private(i,u_tmp,v_tmp,cosp,sinp,tmp_re,tmp_im,\
		ngp_u,ngp_v,loopidx_u,loopidx_v,i_u,i_v,kernel,pcenter,cwgt_re,cwgt_im,flatidx,visidx) 
		for(i = 0; i < visspace.nrows; ++i) {

			// loop over the local uv grid with the gridding kernel
			// process nufft->wsup points on each side of the NGP
			u_tmp = visspace.uv[3*i+0]*visspace.channels[ch_vis];
			v_tmp = -visspace.uv[3*i+1]*visspace.channels[ch_vis];
			//double w_tmp = nufft->uv[3*i+2]*nufft->channels[ch_vis];
			cosp = cos(TWO_PI*(u_tmp*gspace.gcx + v_tmp*gspace.gcy));
			sinp = sin(TWO_PI*(u_tmp*gspace.gcx + v_tmp*gspace.gcy));
			ngp_u = round(u_tmp/gspace.du);
			ngp_v = round(v_tmp/gspace.dv);
			for(loopidx_u = ngp_u - wsup; loopidx_u <= ngp_u + wsup; ++loopidx_u)
			for(loopidx_v = ngp_v - wsup; loopidx_v <= ngp_v + wsup; ++loopidx_v) {

				// kernel and weights 
				// Also factor to center phases so that (0,0) 
				// is in the middle, rather than the corner
				// Use a factor of 0.5 so that we don't double-count 
				// when forming cos() from exp(i*...) 
				i_u = loopidx_u;
				i_v = loopidx_v;
				kernel = kb_g(u_tmp/gspace.du-i_u, v_tmp/gspace.dv-i_v, kb_beta, wsup);
				pcenter = 1-2*((i_u+i_v)&1);
				cwgt_re = 0.5*kernel*pcenter; 
				cwgt_im = 0.5*kernel*pcenter; 

				// Wrap. Beware of aliasing, the de-apodization kernel does not like it 
				while(i_u < 0) i_u += gspace.nu; 
				while(i_v < 0) i_v += gspace.nv; 
				while(i_u >= gspace.nu) i_u -= gspace.nu;
				while(i_v >= gspace.nv) i_v -= gspace.nv;

				// fold since we are using the r2c transform
				if(i_v >= gspace.half_nv) {
					i_u = (gspace.nu - i_u) % gspace.nu; // modulo to preserve i_u = 0
					i_v = gspace.nv - i_v; 
					cwgt_im *= -1;
				} 

				// we have the weights, now grid or de-grid
				flatidx = gspace.half_nv*i_u + i_v; // TODO: properly index with multiple channels/stokes
				visidx = visspace.nrows*visspace.nchannels*stokes+visspace.nrows*ch_vis+i;
				if(direction == FORWARD) {
					tmp_re = cwgt_re*grid[2*flatidx+0];
					tmp_im = cwgt_im*grid[2*flatidx+1];
#pragma omp atomic
					vis[2*visidx+0] += 2*(cosp*tmp_re + sinp*tmp_im);
#pragma omp atomic
					vis[2*visidx+1] += 2*(-sinp*tmp_re + cosp*tmp_im);
				}
				else if(direction == ADJOINT) {
					if(i_v == 0) {
						// Needed for the r2c transform, only in the adjoint direction, why??
						cwgt_re *= 2;
						cwgt_im *= 2;
					}
					tmp_re = vis[2*visidx+0];
					tmp_im = vis[2*visidx+1];
#pragma omp atomic
					grid[2*flatidx+0] += cwgt_re*(cosp*tmp_re - sinp*tmp_im);
#pragma omp atomic
					grid[2*flatidx+1] += cwgt_im*(sinp*tmp_re + cosp*tmp_im);
				}
			}
		}
	}
}

