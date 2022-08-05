#ifndef _DEFLECT_H_
#define _DEFLECT_H_



typedef struct {

	// parameters
	double b;
	double th;
	double f;
	double x, y;
	double rc;
	double qh;
	double ss;
	double sa;
	double z;

	// angular diameter distances
	// pre-computed pars
	double d_l, d_s, d_ls;
	double sigma_c;
	double sin_th, cos_th;
	double sin_sa, cos_sa;

} parametric_lens;



































#endif // _DEFLECT_H_

