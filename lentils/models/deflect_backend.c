
#include <math.h>
#include <stdio.h>
#include <complex.h>
#include <string.h>
#include "common.h"
#include "deflect_backend.h"

void deflect_SIE(parametric_lens lens, double x, double y, double *dx);
void deflect_PEMD_series(parametric_lens lens, double x, double y, double *dx, int want_deriv, double *deriv);
void external_shear(parametric_lens lens, double x, double y, double *ds, int want_deriv, double *deriv);

void deflect_points(parametric_lens lens, double *p_in, int npoints, double *p_out, int want_deriv, double *deriv) 
{

	int p, d;
	double x, y;
	double alpha[2];
	double dtmp[16];
	lens.sin_th = sin((lens.th*PI/180.0)+PI/2.0);
	lens.cos_th = cos((lens.th*PI/180.0)+PI/2.0);
	lens.sin_sa = sin((lens.sa*PI/180.0)+PI/2.0);
	lens.cos_sa = cos((lens.sa*PI/180.0)+PI/2.0);

	for(p = 0; p < npoints; ++p) {
		x = p_in[2*p+0];
		y = p_in[2*p+1];
		memset(alpha, 0, sizeof(alpha));
		if(want_deriv)
			memset(dtmp, 0, sizeof(dtmp));

		// TODO: why is deflect_SIE different?
		// TODO: normalize to Einstein radius
		//deflect_SIE(lens, x, y, dx);
		deflect_PEMD_series(lens, x, y, alpha, want_deriv, dtmp);

		// Calculate the deflection angle for the external shear
		external_shear(lens, x, y, alpha, want_deriv, dtmp);

		// Lens equation 
		p_out[2*p+0] = p_in[2*p+0] - alpha[0]; 
		p_out[2*p+1] = p_in[2*p+1] - alpha[1]; 

		// Derivatives
		if(want_deriv) {
			for(d = 0; d < 8; ++d) {
				deriv[2*(d*npoints+p)+0] = dtmp[2*d+0];
				deriv[2*(d*npoints+p)+1] = dtmp[2*d+1];
			}
		}
	}
}


/*----- Define the function for the deflection angle of the external shear----------------*/
void external_shear(parametric_lens lens, double x, double y, double *ds, int want_deriv, double *deriv)
{
	double cs2,sn2;
	double sx, sy;

	// shear
	sx = x - lens.x;
	sy = y - lens.y;
	cs2 = lens.cos_sa*lens.cos_sa - lens.sin_sa*lens.sin_sa; // cos(2*sa);
	sn2 = 2*lens.sin_sa*lens.cos_sa;  // sin(2*sa);
	ds[0] += lens.ss*(cs2*sx+sn2*sy);
	ds[1] += lens.ss*(sn2*sx-cs2*sy);

	// derivatives
	if(want_deriv) {
		deriv[6] += -lens.ss*cs2; // dalpha_dxl, dalpha_dyl
		deriv[7] += -lens.ss*sn2;
		deriv[8] += -lens.ss*sn2;
		deriv[9] += lens.ss*cs2;
		deriv[12] += cs2*sx+sn2*sy; // dalpha_dss
		deriv[13] += sn2*sx-cs2*sy; 
		deriv[14] += M_PI/180.0*lens.ss*(2*sy*cs2 - 2*sx*sn2); // dalpha_dsa
		deriv[15] += M_PI/180.0*lens.ss*(2*sx*cs2 + 2*sy*sn2);
	}
}




double complex hyp2f1_series(double t, double q, double complex z, int want_deriv, double complex *derivs)
{
	// Computes the Hypergeometric function numerically
	// according to the recipe in O'Riordan et al (Paper III)
	// could also be done using gsl_sf_hyperg_2F1_conj

	int n, max_terms;
	double x, y, r2, q_, a_n, da_dt, ntf;
	double complex f_, u, u_n, u_n_1, w2;
	double complex df_du, df_dt, du_dw2, dw2_dq, dw2_dx, dw2_dy, df_dw2;

	// U and q form Conor
	x = creal(z);
	y = cimag(z);
	r2 = q*q*x*x + y*y;
	q_ = (1 - q*q) / (q*q);
	w2 = q_*r2/(z*z);
	u = 0.5 * (1.0 - csqrt(1.0 - w2));

	// do the series
	f_ = 0.0 + 0.0*I;
	u_n = 1.0 + 0.0*I;
	a_n = 1.0;

	// derivatives
	if(want_deriv) {
		df_du = 0.0 + 0.0*I;
		df_dt = 0.0 + 0.0*I;
		da_dt = 0.0 + 0.0*I;
		u_n_1 = 0.0 + 0.0*I;
	}

	// tests show we hit machine precision around 20-25 iterations 
	max_terms = 20;
	for(n = 0; n < max_terms; n++){
		f_ += a_n * u_n;
		ntf = 2*n + 4 - t;
		if(want_deriv) {
			df_dt += da_dt * u_n;
			df_du += a_n * n * u_n_1; 
			u_n_1 = u_n;
			da_dt = -2*(2 + n)/(ntf*ntf)*a_n + 2*(2 + n - t)/ntf*da_dt;
		}
		u_n *= u;
		a_n *= (ntf-t)/ntf;
	}

	// Derivative chain rule. Do not touch!
	if(want_deriv) {
		du_dw2 = 1/(4*csqrt(1-w2)); 
		df_dw2 = df_du * du_dw2;
		dw2_dq = -2*(q*q*q*q*x*x+y*y)/(q*q*q*z*z);
		dw2_dx = -2*q_*y*(-I*q*q*x+y)/(z*z*z);
		dw2_dy = -2*q_*I*x*(q*q*x+I*y)/(z*z*z);
		derivs[0] = df_dt;
		derivs[1] = df_dw2 * dw2_dq; // df_dq
		derivs[2] = df_dw2 * dw2_dx; // df_dx
		derivs[3] = df_dw2 * dw2_dy; // df_dy
	}
	return f_;
}


/*----- Define the function for the deflection angle of a broken power-law with external shear----------------*/
void deflect_PEMD_series(parametric_lens lens, double xx, double yy, double *d, int want_deriv, double *deriv)
{

	double x, y, rs, rs2, b, t, q;
	double complex A, F, alpha, z, crot;
	double complex fderivs[4];
	double complex dF_dt, dF_dq, dF_dx, dF_dy;
	double complex dA_db, dA_dt, dA_dq, dA_dx, dA_dy;
	double complex dalpha_db, dalpha_dt, dalpha_dq, dalpha_dxs, dalpha_dys;
	double complex dalpha_dx, dalpha_dy, dalpha_dth;

	// Setup
	b = lens.b;
	t = 2*lens.qh; 
	q = lens.f;
	crot = lens.cos_th - I*lens.sin_th; 
	z = ((xx-lens.x) + I*(yy-lens.y)) * crot;
	x = creal(z);
	y = cimag(z);
	rs2 = q*q*x*x+y*y;
	rs = sqrt(rs2);

	// Radial part
	A = b*b/(q*z)*pow(b/rs,t-2);

	// Hypergeometric function
	F = hyp2f1_series(t, q, z, want_deriv, fderivs);

	// Rotate the components (now back into cartesian coords)
	alpha = conj(A*F*crot);
	d[0] += creal(alpha);
	d[1] += cimag(alpha);

	// compute derivatives of parameters only if requested
	if(want_deriv) {
	
		// Derivatives. Do not touch!
		dF_dt = fderivs[0];
		dF_dq = fderivs[1];
		dF_dx = fderivs[2];
		dF_dy = fderivs[3];
		dA_db = A*t/b; 
		dA_dt = A*log(b/rs);
		dA_dq = -A*((t-1)*q*q*x*x+y*y)/(q*rs2);
		dA_dx = -A*(y*y+q*q*x*(x*(t-1)+I*(t-2)*y))/(z*rs2); 
		dA_dy = -A*(I*q*q*x*x+y*((t-2)*x+I*(t-1)*y))/(z*rs2);

		// chain rule, conj, and rotations
		dalpha_db = conj((F*dA_db)*crot);
		dalpha_dt = 2*conj((F*dA_dt + A*dF_dt)*crot); // factor of 2 for t->qh
		dalpha_dq = conj((F*dA_dq + A*dF_dq)*crot);
		dalpha_dxs = conj((F*dA_dx + A*dF_dx)*crot); // derivative w.r.t. shifted and rotated lens coordinates
		dalpha_dys = conj((F*dA_dy + A*dF_dy)*crot);
		dalpha_dx = -(dalpha_dxs*lens.cos_th - dalpha_dys*lens.sin_th); // Jacobian 
		dalpha_dy = -(dalpha_dxs*lens.sin_th + dalpha_dys*lens.cos_th);
		dalpha_dth = M_PI/180*(dalpha_dxs*y - dalpha_dys*x + I*alpha); // rotation
		deriv[0] += creal(dalpha_db);
		deriv[1] += cimag(dalpha_db);
		deriv[2] += creal(dalpha_dt);
		deriv[3] += cimag(dalpha_dt);
		deriv[4] += creal(dalpha_dq);
		deriv[5] += cimag(dalpha_dq);
		deriv[6] += creal(dalpha_dx);
		deriv[7] += cimag(dalpha_dx);
		deriv[8] += creal(dalpha_dy);
		deriv[9] += cimag(dalpha_dy);
		deriv[10] += creal(dalpha_dth); 
		deriv[11] += cimag(dalpha_dth); 
	}
}


/*----- Define the function for the deflection angle of the SIS/SIE or SPMEDs + external shear----------------*/
void deflect_SIE(parametric_lens lens, double x, double y, double *dx)
{
	double sx,sy;
	double sx_r,sy_r;
	double dx_r,dy_r;
	double psi;
	
	// get the rotated source coordinates and psi
	sx = x - lens.x;
	sy = y - lens.y;
	sx_r = sx*lens.cos_th + sy*lens.sin_th;
	sy_r = -sx*lens.sin_th + sy*lens.cos_th;
	psi = sqrt(lens.f*lens.f*(lens.rc*lens.rc + sx_r*sx_r) + sy_r*sy_r);

	//Calculate the deflection angle for the SIE lens
	dx_r = (lens.b*sqrt(lens.f)/sqrt(1.0-lens.f*lens.f))*atan( sqrt(1.0-lens.f*lens.f)*sx_r/(psi+lens.rc));
	dy_r = (lens.b*sqrt(lens.f)/sqrt(1.0-lens.f*lens.f))*atanh(sqrt(1.0-lens.f*lens.f)*sy_r/(psi+lens.rc*lens.f*lens.f));

	//rotate back
	dx[0] = dx_r*lens.cos_th - dy_r*lens.sin_th;
	dx[1] = dx_r*lens.sin_th + dy_r*lens.cos_th;


}

#if 0

	// TODO: Calculate the deflection angle for SPEMDs from fastell by Barkana
	//else
	//{
		//rc = (lenses.rc*lenses.rc);
		//b =  ( 1.5 - lenses.qh) * (lenses.b/(2.0*sqrt(lenses.f)));

		//fastelldefl_(&sx_r,&sy_r,&b,&lenses.qh,&lenses.f,&rc,dfl);

		//dx_tmp = dfl[0];
		//dy_tmp = dfl[1];
	//}
	


/*----- Define the function for the deflection angle of exponential disk (from Keeton with fitting weights by Matt Auger)----------------*/
void deflect_expdisk(Lens lenses, double x, double y, double *d)
{
	int i;
	double sx_r,sy_r;
	double cs,sn,dx,dy;
	double psi,dx_tmp=0.0,dy_tmp=0.0;
	double prefix,denom;
	double rs,k0;
	//external shear
	double ds[2];
	double k[11] = {0.0581345,  -0.02912989,  0.2032523,  -0.29028785,  3.9356616,  -7.37252055, 4.45267609,  0.14064268, -0.54119909,  0.70468137, -0.26382873};
	double s[11] = { 0.04886304,  0.18418226,  0.25150105,  0.76876004,  0.98477198,  1.13834378, 1.27406832,  3.43975013,  6.25407105,  6.98710788,  7.59691785};

	//initialize d to zero
	d[0] = 0.0;
	d[1] = 0.0;

	rotate_sys(lenses, x, y, &sx_r, &sy_r, &sn, &cs, &psi);

	for(i = 0; i < 11; i++){
		rs = lenses.rc*s[i];
		k0 = lenses.b*k[i];

		psi = sqrt(pow(lenses.f,2.0)*(pow(rs,2.0) + pow(sx_r,2.0)) + pow(sy_r,2.0));
		prefix = 2.0*k0*rs*rs*rs/lenses.f;
		denom = pow(psi+rs,2.0)+(1.0-lenses.f*lenses.f)*sx_r*sx_r;

		dx_tmp += prefix*lenses.f*sx_r*(psi+rs*lenses.f*lenses.f)/(rs*psi*denom);
		dy_tmp += prefix*lenses.f*sy_r*(psi+rs)/(rs*psi*denom);

	}

	//rotate back
	dx = dx_tmp*cs - dy_tmp*sn;
	dy = dx_tmp*sn + dy_tmp*cs;

	//Calculate the deflection angle for the external shear
	external_shear(lenses, x, y, ds);

	d[0] = dx + ds[0];
	d[1] = dy + ds[1];
}

/*------Define the function for the deflection angle of a truncated power-law substructure----------------*/
void deflect_powerlaw(Lens lenses, Gdat *sr, int ns, double x, double y, double *d, double *dalpha, int deriv)
{
	double sx_r,sy_r,psi;
	double dr_tmp,dx_tmp,dy_tmp;
	double sigma_c;
	double dls,dl,ds;
	double rt,rb;
	double rs;
	int i;

	//initialize d to zero
	d[0] = 0.0;
	d[1] = 0.0;

	//source angular diameter distance in Kpc
	ds = Cl / (H_0*(1.0+sr->z))*qromb(angular_distance,0.0, sr->z);
	//lens angular diameter distance
	dl = Cl / (H_0*(1.0+lenses.zsub[ns]))*qromb(angular_distance,0.0, lenses.zsub[ns]);
	//lens-source angular diameter distance in Kpc
	dls = Cl / (H_0*(1.0+sr->z))*qromb(angular_distance,lenses.zsub[ns], sr->z);

	//projected distance relative to the substructure in arcseconds
	sx_r = (x - lenses.xsub[ns]);
	sy_r = (y - lenses.ysub[ns]);
	psi = sqrt(sx_r*sx_r+sy_r*sy_r);

	//projected distance relative to lens in arcseconds
	rs = sqrt(pow(lenses.x0-lenses.xsub[ns],2.0)+pow(lenses.y0-lenses.ysub[ns],2.0));

	//critical density surface in 10^10 Msun arcsec^-2
	sigma_c = (Cl*Cl*ds/(4.0*M_PI*G*1.0e10*M_sun*dl*dls))*(dl/206265.0)*(dl/206265.0);

	rb = pow(lenses.msub[ns]*sqrt(6.0*lenses.b)/(rs*sigma_c),2.0/3.0)/M_PI;
	rt = rs*sqrt(M_PI*rb/(6.0*lenses.b));

	//deflection angle
	dr_tmp = (rb *( rt + psi - sqrt(rt*rt+psi*psi)))/psi;

	if(psi >= 1.0e-10){
		dx_tmp = dr_tmp * ( sx_r / psi );
		dy_tmp = dr_tmp * ( sy_r / psi );
	}
	else{
		dx_tmp = 0.0;
		dy_tmp = 0.0;
	}

	//deflection angle in arcseconds
	d[0] = dx_tmp;
	d[1] = dy_tmp;

	if(deriv == 1){
		//deflection angle derivatives
		dalpha[0] = rb/pow(psi,4.) * ( (rt+psi-sqrt(rt*rt+psi*psi))*(psi*psi-2.0*sx_r*sx_r)+ sx_r*sx_r*psi*psi*(1.0/psi - 1.0/sqrt(rt*rt+psi*psi)) );
		dalpha[1] = rb*sx_r*sy_r/pow(psi,4.) * ( psi*psi*(1.0/psi - 1.0/sqrt(rt*rt+psi*psi)) -2.0 *(rt+psi-sqrt(rt*rt+psi*psi)) );
		dalpha[2] = dalpha[1];
		dalpha[3] = rb/pow(psi,4.) * ( (rt+psi-sqrt(rt*rt+psi*psi))*(psi*psi-2.0*sy_r*sy_r)+ sy_r*sy_r*psi*psi*(1.0/psi - 1.0/sqrt(rt*rt+psi*psi)) );
	}
	else{
		for(i = 0; i < 4; i++){
			dalpha[i] = 0.0;
		}
	}
}

/*------Define the function for the deflection angle of a spherical NFW substructure----------------*/
void deflect_NFW(Lens lenses, Gdat *sr, int ns, double x, double y, double *d, double *dalpha, int deriv)
{
	int i;
	double psi,sx_r,sy_r;
	double omega;
	double xx,ks;
	double sigma_c;
	double rhos,rs;
	double dls,dl,ds;
	double cvir,Rvir,Dvir;
	double Fx = 1.0, dr_tmp,dx_tmp,dy_tmp;

	//initialize d to zero
	d[0] = 0.0;
	d[1] = 0.0;

	omega = omega_m*pow(1+lenses.zsub[ns],3)/(omega_m*pow(1+lenses.zsub[ns],3)+omega_l0);
	Dvir = (18.0*M_PI*M_PI+82.0*(omega-1.0)-39.0*(omega-1.0)*(omega-1.0))/omega;

	//compute virial concentration according to Maccio+08 & Duffy+08
	cvir = 0.0;
	if(lenses.shapesub[ns] == 1){
		cvir = 9.23* pow( 0.5*0.01 * lenses.msub[ns] * hh ,-0.091)*pow(1.0+lenses.zsub[ns],-0.71);
	}
	//input cvir
	else if(lenses.shapesub[ns] == 2){
		cvir = lenses.csub[ns];
	}

	//compute virial radius using Bullock+01 in kpc
	Rvir = pow(1.0e10*lenses.msub[ns]*hh*3/(4*M_PI*Dvir*omega_m*rho_c0),0.333333)*1000./hh/(1.0+lenses.zsub[ns]);

	//compute scaling radius in kpc
	rs = Rvir/cvir;

	//compute density normalization in kg kpc-3
	rhos = lenses.msub[ns] /(4.0*M_PI*rs*rs*rs*(log(1.0+cvir) - cvir/(1.0+cvir)));

	//source angular diameter distance in Kpc
	ds = Cl / (H_0*(1.0+sr->z))*qromb(angular_distance,0.0, sr->z);
	//lens angular diameter distance
	dl = Cl / (H_0*(1.0+lenses.zsub[ns]))*qromb(angular_distance,0.0, lenses.zsub[ns]);
	//lens-source angular diameter distance in Kpc
	dls = Cl / (H_0*(1.0+sr->z))*qromb(angular_distance,lenses.zsub[ns], sr->z);

	//critical density surface in 10^10 Msun kpc^-2
	sigma_c = Cl*Cl*ds/(4.0*M_PI*G*1.0e10*M_sun*dl*dls);

	ks = rhos*rs/sigma_c;
	rs = rs/dl*(206264.806);

	//projected distance relative to the substructure in arcseconds
	sx_r = (x - lenses.xsub[ns]);
	sy_r = (y - lenses.ysub[ns]);
	psi = sqrt(sx_r*sx_r+sy_r*sy_r);

	xx = psi/rs;

	if( xx > 1.0 ){
		Fx = 1.0*pow(xx*xx-1.0,-0.5)*atan(pow(xx*xx-1.0,0.5));
	}
	if( xx < 1.0 ){
		Fx = 1.0*pow(1.0-xx*xx,-0.5)*atanh(pow(1.0-xx*xx,0.5));
	}
	if( xx == 1 ){
		Fx = 1.0;
	}

	//deflection angle
	dr_tmp = 4.0*ks*rs*( log(0.5*xx) + Fx ) / xx;

	//project along a and y axes
	if(psi >= 1.0e-10){
		dx_tmp = dr_tmp * ( sx_r / psi );
		dy_tmp = dr_tmp * ( sy_r / psi );
	}
	else{
		dx_tmp = 0.0;
		dy_tmp = 0.0;
	}

	//deflection angle in arcseconds
	d[0] = dx_tmp;
	d[1] = dy_tmp;

	if(deriv == 1){
		ERROR("Derivatives for the NFW defelction angle are not implemented yet");
	}
	else{
		for(i = 0; i < 4; i++){
			dalpha[i] = 0.0;
		}
	}
}

/*------Define the function for the deflection angle----------------*/
void deflect(Lens lenses, Gdat *sr, int ns, double x, double y, double *d)
{
	double dalpha[4]={0.0,0.0,0.0,0.0};

	if (ns == -1){//main lens
		if(lenses.shape == 0){//elliptical power-law, SIS, SIE
			deflect_SIE(lenses,x,y,d);
		}
		else if (lenses.shape == 1){//exponential disk
			deflect_expdisk(lenses,x,y,d);
		}
	}
	else{
		if(lenses.shapesub[ns] == 0){//power-law model for substructure
			deflect_powerlaw(lenses,sr,ns,x, y, d, dalpha, 0);
		}
		else if(lenses.shapesub[ns] == 1 || lenses.shapesub[ns] == 2){//NFW model for substructure
			deflect_NFW(lenses,sr,ns,x,y,d,dalpha,0);
		}
	}
}

/*------Define the function for the linear-correction grid----------------*/
void deflect_grid(Gdat *g,double xx,double yy,double *dg)
{
	double gpot_dx,gpot_dy,t,u,x2;
	double dgdx,dgdy;
	double dy1dx,dy2dx,dy3dx,dy4dx;
	double dy1dy,dy2dy,dy3dy,dy4dy;
	int i1,j1,i2;

	//initialize dg to zero
	dg[0]=0.0;
	dg[1]=0.0;

	gpot_dx=(g->xmax-g->xmin)/(g->dim1);
	gpot_dy=(g->ymax-g->ymin)/(g->dim2);

	i1=(int)(floor((xx-g->xmin)/gpot_dx));
	j1=(int)(floor((yy-g->ymin)/gpot_dy));

	t=xx-(i1*gpot_dx+g->xmin);
	u=yy-(j1*gpot_dy+g->ymin);

	//snap to the nearest pixel if very close
	corr_x_i(t,i1,&x2,&i2);
	t=x2;
	i1=i2;

	corr_x_i(u,j1,&x2,&i2);
	u=x2;
	j1=i2;

	//divide by pixelscale
	t = t / gpot_dx;
	u = u / gpot_dy;

	dgdx=0.0;
	dgdy=0.0;
	//if pixel is inside the gpot grid, then continue
	if(i1>=1 && i1<g->dim1-2){
		if(j1>=1 && j1<g->dim2-2){

			//determine the fraction of pixels
			t=xx-(i1*gpot_dx+g->xmin);
			u=yy-(j1*gpot_dy+g->ymin);

			//divide by pixelscale
			t = t / gpot_dx;
			u = u / gpot_dy;

			//dpot on grid points enclosing the pixel (x,y)
			dy1dx = (g->dpot[i1+1+j1*g->dim1]-g->dpot[i1-1+j1*g->dim1])/(2.0*gpot_dx);
			dy2dx = (g->dpot[i1+2+j1*g->dim1]-g->dpot[i1+j1*g->dim1])/(2.0*gpot_dx);
			dy3dx = (g->dpot[i1+2+(j1+1)*g->dim1]-g->dpot[i1+(j1+1)*g->dim1])/(2.0*gpot_dx);
			dy4dx = (g->dpot[i1+1+(j1+1)*g->dim1]-g->dpot[i1-1+(j1+1)*g->dim1])/(2.0*gpot_dx);

			dy1dy = (g->dpot[i1+(j1+1)*g->dim1]-g->dpot[i1+(j1-1)*g->dim1])/(2.0*gpot_dy);
			dy2dy = (g->dpot[i1+1+(j1+1)*g->dim1]-g->dpot[i1+1+(j1-1)*g->dim1])/(2.0*gpot_dy);
			dy3dy = (g->dpot[i1+1+(j1+2)*g->dim1]-g->dpot[i1+1+j1*g->dim1])/(2.0*gpot_dy);
			dy4dy = (g->dpot[i1+(j1+2)*g->dim1]-g->dpot[i1+j1*g->dim1])/(2.0*gpot_dy);

			//bilinear interpolation
			dgdx = (1.0-t)*(1.0-u)*dy1dx + t*(1.0-u)*dy2dx+t*u*dy3dx + (1.0-t)*u*dy4dx;
			dgdy = (1.0-t)*(1.0-u)*dy1dy + t*(1.0-u)*dy2dy+t*u*dy3dy + (1.0-t)*u*dy4dy;

		}
	}
	dg[0]=dgdx;
	dg[1]=dgdy;
}

/*------Define the function for the last change in the linear-correction grid----------------*/
void deflect_dpot(Gdat *g,double *dpot,double xx,double yy,double *dg)
{
	double gpot_dx,gpot_dy,t,u,x2;
	double dgdx,dgdy;
	double dy1dx,dy2dx,dy3dx,dy4dx;
	double dy1dy,dy2dy,dy3dy,dy4dy;
	int i1,j1,i2;

	//initialize dg to zero
	dg[0]=0.0;
	dg[1]=0.0;

	gpot_dx = (g->xmax-g->xmin)/(g->dim1);
	gpot_dy = (g->ymax-g->ymin)/(g->dim2);

	i1 = (int)(floor((xx-g->xmin)/gpot_dx));
	j1= (int)(floor((yy-g->ymin)/gpot_dy));

	t = xx-(i1*gpot_dx+g->xmin);
	u = yy-(j1*gpot_dy+g->ymin);

	//snap to the nearest pixel if very close
	corr_x_i(t,i1,&x2,&i2);
	t = x2;
	i1 = i2;

	corr_x_i(u,j1,&x2,&i2);
	u = x2;
	j1 = i2;

	//divide by pixlescale
	t = t / gpot_dx;
	u = u / gpot_dy;

	//if pixel is inside the gpot grid, then continue
	dgdx = 0.0;
	dgdy = 0.0;

	if(i1 >= 1 && i1 < g->dim1-2){
		if(j1 >= 1 && j1 < g->dim2-2){

			//dpot on grid points enclosing the pixel (x,y)
			dy1dx = (dpot[i1+1+j1*g->dim1]-dpot[i1-1+j1*g->dim1])/(2.0*gpot_dx);
			dy2dx = (dpot[i1+2+j1*g->dim1]-dpot[i1+j1*g->dim1])/(2.0*gpot_dx);
			dy3dx = (dpot[i1+2+(j1+1)*g->dim1]-dpot[i1+(j1+1)*g->dim1])/(2.0*gpot_dx);
			dy4dx = (dpot[i1+1+(j1+1)*g->dim1]-dpot[i1-1+(j1+1)*g->dim1])/(2.0*gpot_dx);

			dy1dy = (dpot[i1+(j1+1)*g->dim1]-dpot[i1+(j1-1)*g->dim1])/(2.0*gpot_dy);
			dy2dy = (dpot[i1+1+(j1+1)*g->dim1]-dpot[i1+1+(j1-1)*g->dim1])/(2.0*gpot_dy);
			dy3dy = (dpot[i1+1+(j1+2)*g->dim1]-dpot[i1+1+j1*g->dim1])/(2.0*gpot_dy);
			dy4dy = (dpot[i1+(j1+2)*g->dim1]-dpot[i1+j1*g->dim1])/(2.0*gpot_dy);

			//bilinear extrapolation
			dgdx = (1.0-t)*(1.0-u)*dy1dx + t*(1.0-u)*dy2dx+t*u*dy3dx + (1.0-t)*u*dy4dx;
			dgdy = (1.0-t)*(1.0-u)*dy1dy + t*(1.0-u)*dy2dy+t*u*dy3dy + (1.0-t)*u*dy4dy;

		}
	}
	dg[0] = dgdx;
	dg[1] = dgdy;

}

/*----Define the function for the lens convergence----------------*/
double convergence_lens(Lens lenses, double x, double y)
{
	double cs,sn,sx_r,sy_r,psi,psi_dmy;
	double kappa_tmp = 0.0, b;

	rotate_sys(lenses, x, y, &sx_r, &sy_r, &sn, &cs, &psi_dmy);

	//elliptical power-law, SIS, SIE
	if(lenses.shape == 0){

		if(lenses.qh == 0.5){//SIE,SIS
			psi = sqrt(pow(lenses.f,2)*(pow(lenses.rc,2)+pow(sx_r,2))+pow(sy_r,2));
			kappa_tmp = 0.5*lenses.b*sqrt(lenses.f)/psi;
		}
		else{//SPEMD
			psi = pow(lenses.rc,2) + pow(sx_r,2) + pow(sy_r/lenses.f,2);
			b =  ( 1.5 - lenses.qh) * (lenses.b/(2.0*sqrt(lenses.f)));
			kappa_tmp = 0.5*b*pow(psi,-1.0*lenses.qh);
		}
	}//exponential disk
	else if(lenses.shape == 1){
		kappa_tmp = 0.0;
		ERROR("you have not defined kappa for an exponential disk yet");
	}

	return kappa_tmp;
}

/*----Define the function for teh convergence of the potential corrections----------------*/
void convergence_gpot(Gdat *g, Gdat *kappa)
{
	double gpot_dx,gpot_dy;
	double tmp_x,tmp_y;
	int i,j;

	gpot_dx = (g->xmax - g->xmin) / (g->dim1);
	gpot_dy = (g->ymax - g->ymin) / (g->dim2);

	for( i = 1; i < g->dim1-1; i++ ){
		for( j = 1; j < g->dim2-1; j++ ){
			kappa->mask[i+j*g->dim1] = 1.0;
			if( g->mask[i-1+j*g->dim1] == 1 && g->mask[i+j*g->dim1] == 1 && g->mask[i+1+j*g->dim1] == 1 && g->mask[i+(j-1)*g->dim1] == 1 && g->mask[i+(j+1)*g->dim1] == 1 ){

				tmp_x = (0.5/(pow(gpot_dx,2)))*(g->dpot[i-1+j*g->dim1] - 2.0*g->dpot[i+j*g->dim1] + g->dpot[i+1+j*g->dim1]);
				tmp_y = (0.5/(pow(gpot_dy,2)))*(g->dpot[i+(j-1)*g->dim1] - 2.0*g->dpot[i+j*g->dim1] + g->dpot[i+(j+1)*g->dim1]);

				//potential correction convergence
				kappa->dpot[i+j*g->dim1] = tmp_x+tmp_y;
				kappa->mask[i+j*g->dim1] = 1.0;
			}
		}
	}
}


#endif
