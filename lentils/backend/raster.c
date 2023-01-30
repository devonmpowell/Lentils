


#include <math.h>
#include <string.h>
#include <stdio.h>
#include "common.h"


typedef struct {
	double verts[3][2]; // 3 vertex positions
	double deriv[3][3][2][2]; // derivatives of these verts wrt. the original triangle verts 
	int idxmin[2];
	int idxmax[2];
} tri_info;

#if 0
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
#endif


double reduce_triangle(tri_info *tri_in) {
	double x1, y1, x2, y2, x3, y3; 
	x1 = tri_in->verts[0][0]; 
	y1 = tri_in->verts[0][1];
	x2 = tri_in->verts[1][0];
	y2 = tri_in->verts[1][1];
	x3 = tri_in->verts[2][0];
	y3 = tri_in->verts[2][1];


	int i;
	printf("Reduce derivs = ");
	for (i = 0; i < 12; ++i) {
		printf("%.2e ", ((double*)tri_in->deriv)[i]);
	}
	printf("\n");

	//return 0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2));
	return -0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2));

	// TODO: derivatives
}


void split_edge(double *x0, double *dx0_dxp, double w0, double *x1, double *dx1_dxp, double w1, int spax, double *vout, double *dvout_dxp) {

	int i, j, p, snax;

	double w[2];
	double wnorm = 1.0/(w0+w1);
	w[0] = w0*wnorm;
	w[1] = w1*wnorm;
	vout[0] = w[0]*x0[0] + w[1]*x1[0];
	vout[1] = w[0]*x0[1] + w[1]*x1[1];

	// non-split axis
	snax = (spax+1)%2;

	// TODO: derivatives
	double dw_dx0[2][2];
	double dw_dx1[2][2];
	dw_dx0[0][snax] = 0.0; 
	dw_dx0[1][snax] = 0.0; 
	dw_dx0[0][spax] = w[0]*wnorm;
	dw_dx0[1][spax] = -w[0]*wnorm;
	dw_dx1[0][snax] = 0.0; 
	dw_dx1[1][snax] = 0.0; 
	dw_dx1[0][spax] = w[1]*wnorm;
	dw_dx1[1][spax] = -w[1]*wnorm;


	double dwi_dxp[2];


	// dvout_dp = dv_dx0 * dx0_dp + dv_dx1 * dx1_dp
	for(p = 0; p < 3; ++p) {

		//dwi_dxp[0] = dw_dx0[0][0] * dx0_dxp[4*p+2*0+0]
		//dwi_dxp[1] = 



		//for(j = 0; j < 2; ++j) {
			//dvout_dxp[4*p+2*i+j] += dv_dx0[i][j] * dx0_dp[6*i+2*p+j] + dv_dx1[i][j] * dx1_dp[6*i+2*p+j];
		//}
	
	
	}
}


void rasterize_triangle(double *verts_in, double *deriv_in, int nx, int ny, double xmin, double xmax, double ymin, double ymax, double *out) {

	int vbits, i, j, v, v0, v1, s, t, vb;
	int spax, dmax, nstack, siz, spind; 
	double dx, dy, area, spcoord;
	int inds_buffer[3];
	double verts_buffer[5][2];
	double deriv_buffer[5][3][3][2];
	double vdists[3]; 
	tri_info tri; 
	tri_info stack[64]; // only an insane triangle size (>512 pixels across) would overflow

	// arrays encoding the ways that a triangle can be split by a plane
	// Do not touch!
	const int num_new_verts[8] = {0,2,2,2,2,2,2,0};
	const int edge_inds[8][2][2] = {{{0,0},{0,0}},{{0,1},{0,2}},{{0,1},{1,2}},
		{{0,2},{1,2}},{{0,2},{1,2}},{{0,1},{1,2}},{{0,1},{0,2}},{{0,0},{0,0}}};
	const int num_new_tris[8] = {1,3,3,3,3,3,3,1};
	const int new_tri_inds[8][3][3] = {{{0,1,2},{0,0,0},{0,0,0}},{{1,2,4},{1,4,3},{0,3,4}}, 
		{{0,3,4},{0,4,2},{1,4,3}},{{2,3,4},{0,4,3},{0,1,4}},{{0,1,4},{0,4,3},{2,3,4}}, 
		{{1,4,3},{0,3,4},{0,4,2}},{{0,3,4},{1,2,4},{1,4,3}},{{0,1,2},{0,0,0},{0,0,0}}};
	const int new_tri_sides[8][3] = {{0,0,0},{0,0,1},{0,0,1},{0,1,1},{0,0,1},{0,1,1},{0,1,1},{1,0,0}};

	// TODO: return if any parameters are bad 
	// TODO: make function signature cleaner, so we don't have to do this
	dx = (xmax-xmin)/nx;
	dy = (ymax-ymin)/ny;
	double dd[2] = {dx, dy};
	double bmin[2] = {xmin, ymin};

	// set up the original triangle
	// TODO: more careful clamping
	memset(&tri, 0, sizeof(tri));
	tri.idxmin[0] = nx;
	tri.idxmin[1] = ny;
	tri.idxmax[0] = 0;
	tri.idxmax[1] = 0;
	for(v = 0; v < 3; ++v) {
		tri.verts[v][0] = verts_in[2*v+0];
		tri.verts[v][1] = verts_in[2*v+1];
		tri.deriv[v][v][0][0] = 1.0; // Identity matrix for original verts
		tri.deriv[v][v][1][1] = 1.0; 
		i = floor((tri.verts[v][0]-xmin)/dx);
		j = floor((tri.verts[v][1]-ymin)/dy);
		if(i < tri.idxmin[0]) tri.idxmin[0] = i;
		if(j < tri.idxmin[1]) tri.idxmin[1] = j;
		if(i+1 > tri.idxmax[0]) tri.idxmax[0] = i+1;
		if(j+1 > tri.idxmax[1]) tri.idxmax[1] = j+1;
	}

	double area_orig = reduce_triangle(&tri);
	double area_tot = 0.0;
	int max_stack = 0;
	int num_reduce = 0;
	int num_split = 0;
	double reduce_min = 1.0e10;
	double reduce_max = -1.0e10;

	// push the original polyhedron onto the stack
	// and recurse until child polyhedra occupy single rasters
	nstack = 0;
	stack[nstack++] = tri; 
	while(nstack > 0) {

		// pop the stack
		tri = stack[--nstack];
		
		// find the longest axis along which to split 
		for(dmax = 0, spax = 0, i = 0; i < 2; ++i) {
			siz = tri.idxmax[i] - tri.idxmin[i];
			if(siz > dmax) {
				dmax = siz; 
				spax = i;
			}	
		}
		spind = tri.idxmin[spax] + dmax/2;
		spcoord = spind*dd[spax] + bmin[spax];

		// if all three axes are only one raster long, reduce the single raster to the dest grid
		if(dmax == 1) {
			i = tri.idxmin[0];
			j = tri.idxmin[1];
			area = reduce_triangle(&tri);
			out[i*ny+j] += area;

			// TODO: derivs

			if(area < reduce_min) reduce_min = area;
			if(area > reduce_max) reduce_max = area;
			area_tot += area;
			num_reduce += 1;
			continue;
		}

		// set up vertex and index buffers for the split
		// vbits tells us the split geometry,
		// encoding which vertices are on which side of the split
		memset(verts_buffer, 0, sizeof(verts_buffer));
		memset(deriv_buffer, 0, sizeof(deriv_buffer));
		memcpy(verts_buffer, tri.verts, sizeof(tri.verts));
		memcpy(deriv_buffer, tri.deriv, sizeof(tri.deriv));
		for(vbits = 0, v = 0; v < 3; ++v) {
			vdists[v] = verts_buffer[v][spax] - spcoord;
			vbits |= ((vdists[v] > 0.) << v);
		}
		inds_buffer[0] = tri.idxmin[spax];
		inds_buffer[1] = spind;
		inds_buffer[2] = tri.idxmax[spax];

		// interpolate new vertices where needed,
		// add them to the end of the buffer
		for(v = 0; v < num_new_verts[vbits]; ++v) {
			v0 = edge_inds[vbits][v][0];
			v1 = edge_inds[vbits][v][1];
			split_edge(verts_buffer[v0], deriv_buffer[v0][0][0], vdists[v1], 
					verts_buffer[v1], deriv_buffer[v1][0][0], -vdists[v0],
					spax, verts_buffer[3+v], deriv_buffer[3+v][0][0]);
		}

		// set up new triangles after the split
		// push them back on the stack
		for(t = 0; t < num_new_tris[vbits]; ++t) {
			for(v = 0; v < 3; ++v) {
				vb = new_tri_inds[vbits][t][v]; 
				memcpy(tri.verts[v], verts_buffer[vb], 2*sizeof(double));
				memcpy(tri.deriv[v], deriv_buffer[vb], 12*sizeof(double));
			}
			s = new_tri_sides[vbits][t];
			tri.idxmin[spax] = inds_buffer[s];
			tri.idxmax[spax] = inds_buffer[s+1];
			stack[nstack++] = tri;
		}

		num_split += 1;
		if(nstack > max_stack) max_stack = nstack;
	}
	printf("Triangle area total, original, error = %f, %f, %.2e\n", area_tot, area_orig, area_tot/area_orig-1.0);
	printf("areas min max = %f %f\n", reduce_min, reduce_max); 
	printf("tri_info size = %ld bytes\n", sizeof(tri_info)); 
	printf("Max stack depth = %d, total reductions = %d, total splits = %d\n", max_stack, num_reduce, num_split);
}



void manifold_lens_matrix_csr(int im_nx, int im_ny, int ncast, char *mask, 
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
			//triangle_geometry(stmp, pos_tmp, weights, &atmp);
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



