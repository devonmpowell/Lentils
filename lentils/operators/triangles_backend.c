#include <math.h>
#include <stdio.h>
#include "common.h"
#include "triangles_backend.h"

/*------Define the function that check whether a point is inside or outside a triangle of the source grid----------------*/
void triangle_geometry(double *s, double *tri, double *weights_out, double *area)
{

	// compute the barycentric coordinates w[]
	// they sum to 1, they are the interpolation weights
	double c[2],a[2],b[2];
	double w[3], invarea;
	a[0]=tri[2*0+0]-s[0];
	a[1]=tri[2*0+1]-s[1];
	b[0]=tri[2*1+0]-s[0];
	b[1]=tri[2*1+1]-s[1];
	c[0]=tri[2*2+0]-s[0];
	c[1]=tri[2*2+1]-s[1];
	w[0]=b[0]*c[1]-c[0]*b[1];
	w[1]=c[0]*a[1]-a[0]*c[1];
	w[2]=a[0]*b[1]-b[0]*a[1];
	*area = w[0]+w[1]+w[2];
	invarea = 1.0/(*area);
	w[0] *= invarea;
	w[1] *= invarea;
	w[2] *= invarea;
	weights_out[0] = w[0];
	weights_out[1] = w[1];
	weights_out[2] = w[2];
}

/*------Define the function that check whether a point is inside or outside a triangle of the source grid----------------*/
void triangle_geometry_from_inds(double *s, int *tri_inds, double *weights_out, int num_points, double *pos)
{

	int i, p[3];
	double stmp[2]; 
	double atmp;
	double pos_tmp[3*2]; 
	for(i = 0; i < num_points; ++i) {
		stmp[0] = s[2*i+0];
		stmp[1] = s[2*i+1];
		p[0] = tri_inds[3*i+0];
		p[1] = tri_inds[3*i+1];
		p[2] = tri_inds[3*i+2];
		pos_tmp[2*0+0] = pos[2*p[0]+0];
		pos_tmp[2*0+1] = pos[2*p[0]+1];
		pos_tmp[2*1+0] = pos[2*p[1]+0];
		pos_tmp[2*1+1] = pos[2*p[1]+1];
		pos_tmp[2*2+0] = pos[2*p[2]+0];
		pos_tmp[2*2+1] = pos[2*p[2]+1];
		triangle_geometry(stmp, pos_tmp, &weights_out[3*i], &atmp);
		//if(atmp <= 0.0) {
			//weights_out[3*i+0] = 0.0;
			//weights_out[3*i+1] = 0.0;
			//weights_out[3*i+2] = 0.0;
			//continue;
		//}
	}
}


// gradient matrix assembly for csr format
void triangle_gradient_csr(int num_tris, int *tri_inds, double *tri_pos, int *row_inds, int *cols, double *vals)
{

	int i, t, row, v0, v1, v2;
	double p0x, p0y, p1x, p1y, p2x, p2y;
	double weight, area;

	printf("Num tris = %d\n", num_tris);

	for(t = 0, i = 0, row = 0; t < num_tris; ++t) {

		// triangle geometry
		v0 = tri_inds[3*t+0];
		v1 = tri_inds[3*t+1];
		v2 = tri_inds[3*t+2];
		p0x = tri_pos[2*v0+0];
		p0y = tri_pos[2*v0+1];
		p1x = tri_pos[2*v1+0];
		p1y = tri_pos[2*v1+1];
		p2x = tri_pos[2*v2+0];
		p2y = tri_pos[2*v2+1];
		area = 0.5*(p0y*(p1x - p2x) + p1y*p2x - p1x*p2y + p0x*(-p1y + p2y));
		//weight = 1.0;
		weight = 1.0/(2*area);
		weight *= sqrt(fabs(area)); // TODO: figure out this weighting....

		// x-component of the gradient
		row_inds[row++] = i;
		vals[i] = (-p1y + p2y)*weight; cols[i++] = v0;
		vals[i] = (p0y - p2y)*weight; cols[i++] = v1;
		vals[i] = (-p0y + p1y)*weight; cols[i++] = v2;

		// y-component of the gradient
		row_inds[row++] = i;
		vals[i] = (p1x - p2x)*weight; cols[i++] = v0;
		vals[i] = (-p0x + p2x)*weight; cols[i++] = v1;
		vals[i] = (p0x - p1x)*weight; cols[i++] = v2;
	}
	row_inds[row] = i;
}


void delaunay_lens_matrix_csr(int im_nx, int im_ny, int ncast, char *mask, 
		double *uncasted_points, int *uncasted_tri_inds, double *casted_points, int *all_tri_inds, 
		int *row_inds, int *cols, double *vals) {

	int row, i, iim, jim, v, cast_idx, uncast_idx, tind, p[3];
	double atmp, stmp[2], pos_tmp[6], weights[3];
	for(iim = 0, i = 0, row = 0, cast_idx = 0, uncast_idx = 0; iim < im_nx; ++iim)
	for(jim = 0; jim < im_ny; ++jim, ++row) {
		row_inds[row] = i;
		if(!mask[row]) continue;
		if(iim%ncast || jim%ncast) {
			// uncasted points
			stmp[0] = uncasted_points[2*uncast_idx+0];
			stmp[1] = uncasted_points[2*uncast_idx+1];
			tind = uncasted_tri_inds[uncast_idx++];
			if(tind < 0) continue;
			for(v = 0; v < 3; ++v) {
				p[v] = all_tri_inds[3*tind+v];
				pos_tmp[2*v+0] = casted_points[2*p[v]+0];
				pos_tmp[2*v+1] = casted_points[2*p[v]+1];
			
			}
			triangle_geometry(stmp, pos_tmp, weights, &atmp);
			for(v = 0; v < 3; ++v) {
				vals[i] = weights[v]; 
				cols[i++] = p[v];
			}
		}
		else {
			// casted points
			vals[i] = 1.0; 
			cols[i++] = cast_idx++;
		}
	}
	row_inds[row] = i;
}





