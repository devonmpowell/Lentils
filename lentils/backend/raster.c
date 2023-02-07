


#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "deflect.h"

#define STACK_SIZE 64
#define FRAGBUF_SIZE 128000 

typedef struct {
	double verts[3][2]; // 3 vertex positions
	double barys[3][2]; // barycentric coords
	double deriv[3][3][2][2]; // derivatives of these verts wrt. the original triangle verts 
	int idxmin[2];
	int idxmax[2];
} tri_info;

typedef struct {
	double area; 
	//double deriv[3][3][2][2]; // derivatives of these verts wrt. the original triangle verts 
	long grididx;
} fragment; 

int compare_frag_idx(const void* a, const void* b) {
    long idxa = ((fragment*)a)->grididx;
    long idxb = ((fragment*)b)->grididx;
    return (idxa > idxb) - (idxa < idxb); 
}


double reduce_triangle(tri_info *tri_in) {
	double x1, y1, x2, y2, x3, y3; 
	//x1 = tri_in->verts[0][0]; 
	//y1 = tri_in->verts[0][1];
	//x2 = tri_in->verts[1][0];
	//y2 = tri_in->verts[1][1];
	//x3 = tri_in->verts[2][0];
	//y3 = tri_in->verts[2][1];
	//return -0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2));
	

	x1 = tri_in->barys[0][0]; 
	y1 = tri_in->barys[0][1];
	x2 = tri_in->barys[1][0];
	y2 = tri_in->barys[1][1];
	x3 = tri_in->barys[2][0];
	y3 = tri_in->barys[2][1];
	return -0.25*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2));


	//int i;
	//printf("Reduce derivs = ");
	//for (i = 0; i < 12; ++i) {
		//printf("%.2e ", ((double*)tri_in->deriv)[i]);
	//}
	//printf("\n");

	return -0.5*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2));

	// TODO: derivatives
}


void split_edge(double *x0, double *b0, double *dx0_dxp, double w0, 
		double *x1, double *b1, double *dx1_dxp, double w1, int spax, 
		double *vout, double *bout, double *dvout_dxp) {

	int i, j, p, snax;

	double w[2];
	double wnorm = 1.0/(w0+w1);
	w[0] = w0*wnorm;
	w[1] = w1*wnorm;
	vout[0] = w[0]*x0[0] + w[1]*x1[0];
	vout[1] = w[0]*x0[1] + w[1]*x1[1];

	bout[0] = w[0]*b0[0] + w[1]*b1[0];
	bout[1] = w[0]*b0[1] + w[1]*b1[1];

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


void rasterize_triangle(double *verts_in, double *deriv_in, image_space grid, fragment *fragbuffer, int *nfrag) {

	int vbits, i, j, v, v0, v1, s, t, vb;
	int spax, dmax, nstack, siz, spind; 
	double area, spcoord;
	int inds_buffer[3];
	double verts_buffer[5][2];
	double barys_buffer[5][2];
	double deriv_buffer[5][3][3][2];
	double vdists[3]; 
	tri_info tri; 
	tri_info stack[STACK_SIZE]; // only an insane triangle size (>512 pixels across) would overflow

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

	double dd[2] = {grid.dx, grid.dy};
	double bmin[2] = {grid.xmin, grid.ymin};

	// set up the original triangle
	// TODO: more careful clamping
	memset(&tri, 0, sizeof(tri));
	tri.idxmin[0] = grid.nx;
	tri.idxmin[1] = grid.ny;
	tri.idxmax[0] = 0;
	tri.idxmax[1] = 0;
	tri.barys[0][0] = 0.0;
	tri.barys[0][1] = 0.0;
	tri.barys[1][0] = 0.0;
	tri.barys[1][1] = 1.0;
	tri.barys[2][0] = 1.0;
	tri.barys[2][1] = 0.0;
	for(v = 0; v < 3; ++v) {
		tri.verts[v][0] = verts_in[2*v+0];
		tri.verts[v][1] = verts_in[2*v+1];
		tri.deriv[v][v][0][0] = 1.0; // Identity matrix for original verts
		tri.deriv[v][v][1][1] = 1.0; 
		i = floor((tri.verts[v][0]-grid.xmin)/grid.dx);
		j = floor((tri.verts[v][1]-grid.ymin)/grid.dy);
		if(i < tri.idxmin[0]) tri.idxmin[0] = i;
		if(j < tri.idxmin[1]) tri.idxmin[1] = j;
		if(i+1 > tri.idxmax[0]) tri.idxmax[0] = i+1;
		if(j+1 > tri.idxmax[1]) tri.idxmax[1] = j+1;
	}

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

		// if both axes are only one raster long, reduce the fragment to the grid
		if(dmax == 1) {
			i = tri.idxmin[0];
			j = tri.idxmin[1];

			if(i < 0 || i >= grid.nx) continue;
			if(j < 0 || j >= grid.ny) continue;

			// TODO: derivs
			area = reduce_triangle(&tri);
			fragbuffer[*nfrag].area = area;
			fragbuffer[*nfrag].grididx = grid.ny*i + j;
			(*nfrag)++;

			if(*nfrag >= FRAGBUF_SIZE) {
				printf("Warning! Frag buffer overflow!!!\n");
				fflush(stdout);
				return;
			} 


			continue;
		}

		// set up vertex and index buffers for the split
		// vbits tells us the split geometry,
		// encoding which vertices are on which side of the split
		memset(verts_buffer, 0, sizeof(verts_buffer));
		memset(barys_buffer, 0, sizeof(barys_buffer));
		memset(deriv_buffer, 0, sizeof(deriv_buffer));
		memcpy(verts_buffer, tri.verts, sizeof(tri.verts));
		memcpy(barys_buffer, tri.barys, sizeof(tri.barys));
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
			split_edge(verts_buffer[v0], barys_buffer[v0], deriv_buffer[v0][0][0], vdists[v1], 
					verts_buffer[v1], barys_buffer[v1], deriv_buffer[v1][0][0], -vdists[v0],
					spax, verts_buffer[3+v], barys_buffer[3+v], deriv_buffer[3+v][0][0]);
		}

		// set up new triangles after the split
		// push them back on the stack
		for(t = 0; t < num_new_tris[vbits]; ++t) {

			s = new_tri_sides[vbits][t];
			tri.idxmin[spax] = inds_buffer[s];
			tri.idxmax[spax] = inds_buffer[s+1];
			if(tri.idxmax[0] <= 0 || tri.idxmin[0] >= grid.nx) continue;
			if(tri.idxmax[1] <= 0 || tri.idxmin[1] >= grid.ny) continue;
			for(v = 0; v < 3; ++v) {
				vb = new_tri_inds[vbits][t][v]; 
				memcpy(tri.verts[v], verts_buffer[vb], 2*sizeof(double));
				memcpy(tri.barys[v], barys_buffer[vb], 2*sizeof(double));
				memcpy(tri.deriv[v], deriv_buffer[vb], 12*sizeof(double));
			}
			stack[nstack++] = tri;

			if(nstack >= STACK_SIZE) {
				printf("Error! Rasterization stack overflow!\n");
				fflush(stdout);
				return;
			} 
		}
	}
}


void manifold_lens_matrix_csr(image_space imspace, image_space srcspace, global_lens_model lensmodel, 
		int *row_inds, int *cols, double *vals) {

	int row, col, i, iim, jim, v, t, f;
	double val, pxmin, pxmax, pymin, pymax;
	double impos[9][2], srcpos[9][2], tripos[3][2];

	int nfrag;
	fragment *fragbuffer;
	fragbuffer = (fragment*) calloc(FRAGBUF_SIZE, sizeof(fragment));

	for(iim = 0, i = 0, row = 0; iim < imspace.nx; ++iim)
	for(jim = 0; jim < imspace.ny; ++jim, ++row) {
		row_inds[row] = i;
		if(!imspace.mask[row]) continue;

		// four pixel corners, plus the center
		pxmin = imspace.xmin + iim*imspace.dx;
		pxmax = imspace.xmin + (iim+1)*imspace.dx;
		pymin = imspace.ymin + jim*imspace.dy;
		pymax = imspace.ymin + (jim+1)*imspace.dy;

#if 0
		impos[0][0] = pxmin;
		impos[0][1] = pymin;
		impos[1][0] = pxmax;
		impos[1][1] = pymin;
		impos[2][0] = pxmax;
		impos[2][1] = pymax;
		impos[3][0] = pxmin;
		impos[3][1] = pymax;
		impos[4][0] = 0.5*(pxmin+pxmax);
		impos[4][1] = 0.5*(pymin+pymax);
		deflect_points(lensmodel, &impos[0][0], 5, &srcpos[0][0], 0, NULL);

		// split and cast four triangles
		for(t = 0, nfrag = 0; t < 4; ++t) {
			tripos[1][0] = srcpos[t][0];
			tripos[1][1] = srcpos[t][1];
			tripos[0][0] = srcpos[(t+1)%4][0];
			tripos[0][1] = srcpos[(t+1)%4][1];
			tripos[2][0] = srcpos[4][0];
			tripos[2][1] = srcpos[4][1];
			rasterize_triangle(&tripos[0][0], NULL, srcspace, fragbuffer, &nfrag);
			// TODO: refinement
		}
#else
		impos[0][0] = pxmin;
		impos[0][1] = pymin;

		impos[1][0] = 0.5*(pxmin+pxmax);
		impos[1][1] = pymin;

		impos[2][0] = pxmax;
		impos[2][1] = pymin;

		impos[3][0] = pxmax;
		impos[3][1] = 0.5*(pymin+pymax);
		

		impos[4][0] = pxmax;
		impos[4][1] = pymax;

		impos[5][0] = 0.5*(pxmin+pxmax);
		impos[5][1] = pymax;
		

		impos[6][0] = pxmin;
		impos[6][1] = pymax;

		impos[7][0] = pxmin;
		impos[7][1] = 0.5*(pymin+pymax);

		
		impos[8][0] = 0.5*(pxmin+pxmax);
		impos[8][1] = 0.5*(pymin+pymax);
		
		
		
		deflect_points(lensmodel, &impos[0][0], 9, &srcpos[0][0], 0, NULL);

		// split and cast four triangles
		for(t = 0, nfrag = 0; t < 8; ++t) {
			tripos[1][0] = srcpos[t][0];
			tripos[1][1] = srcpos[t][1];
			tripos[0][0] = srcpos[(t+1)%8][0];
			tripos[0][1] = srcpos[(t+1)%8][1];
			tripos[2][0] = srcpos[8][0];
			tripos[2][1] = srcpos[8][1];
			rasterize_triangle(&tripos[0][0], NULL, srcspace, fragbuffer, &nfrag);
			// TODO: refinement
		}



#endif

		//printf("Num fragments = %d\n", nfrag);

		// sort the fragment buffer by column index
		qsort(fragbuffer, nfrag, sizeof(fragment), compare_frag_idx);

		// reduce fragments to each src-plane pixel
		val = 0.0;
		col = fragbuffer[0].grididx;
		for(f = 0; f < nfrag; ++f) {
			if(fragbuffer[f].grididx != col) {
				vals[i] = val; 
				cols[i] = col;
				i++;
				val = 0.0;
				col = fragbuffer[f].grididx;
			}
			else {
				val += fragbuffer[f].area;
			}
		} 
		vals[i] = val; 
		cols[i] = col;
		i++;
	}
	row_inds[row] = i;
	free(fragbuffer);
}



