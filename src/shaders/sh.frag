// Monte Carlo estimator
struct SHSample {
	Vector3d sph;
	Vector3d vec;
	double *coeff;
};

void SH_setup_spherical_samples(SHSample samples[], int sqrt_n_samples)
{
	// fill an N*N*2 array with uniformly distributed
	// samples across the sphere using jittered stratification
	int i=0; // array index
	double oneoverN = 1.0/sqrt_n_samples;
	for(int a=0; a<sqrt_n_samples; a++) {
		for(int b=0; b<sqrt_n_samples; b++) {
			// generate unbiased distribution of spherical coords
			double x = (a + random()) * oneoverN; // do not reuse results
			double y = (b + random()) * oneoverN; // each sample must be random
			double theta = 2.0 * acos(sqrt(1.0 - x));
			double phi = 2.0 * PI * y;
			samples[i].sph = Vector3d(theta,phi,1.0);
			// convert spherical coords to unit vector
			Vector3d vec(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
			samples[i].vec = vec;
			// precompute all SH coefficients for this sample
			for(int l=0; l<n_bands; ++l) {
				for(int m=-l; m<=l; ++m) {
					int index = l*(l+1)+m;
					samples[i].coeff[index] = SH(l,m,theta,phi);
				}
			}
			++i;
		}
	}
}

// evaluate an Associated Legendre Polynomial P(l,m,x) at x
double P(int l, int m, double x)
{
	double pmm = 1.0;
	if(m>0) {
		double somx2 = sqrt((1.0-x)*(1.0+x));
		double fact = 1.0;
		for(int i=1; i<=m; i++) {
			pmm *= (-fact) * somx2;
			fact += 2.0;
		}
	}
	if(l==m) return pmm;

	double pmmp1 = x * (2.0*m+1.0) * pmm;
	if(l==m+1) return pmmp1;

	double pll = 0.0;
	for(int ll=m+2; ll<=l; ++ll) {
		pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m);
		pmm = pmmp1;
		pmmp1 = pll;
	}
	return pll;
}

// Note: the fastest and most accurate way to implement
// factorial(x) is as a table of precalculated floating point values.
// You will never need more than 33 entries in the table.)
// double K(int l, int m)
{
	// renormalisation constant for SH function
	double temp = ((2.0*l+1.0)*factorial(l-m)) / (4.0*PI*factorial(l+m));
	return sqrt(temp);
}

// return a point sample of a Spherical Harmonic basis function
// l is the band, range [0..N]
// m in the range [-l..l]
// theta in the range [0..Pi]
// phi in the range [0..2*Pi]
double SH(int l, int m, double theta, double phi)
{
	const double sqrt2 = sqrt(2.0);
	if(m==0) 
		return K(l,0)*P(l,m,cos(theta));
	else if(m>0) 
		return sqrt2*K(l,m)*cos(m*phi)*P(l,m,cos(theta));
	else return sqrt2*K(l,-m)*sin(-m*phi)*P(l,-m,cos(theta));
}

typedef double (*SH_polar_fn)(double theta, double phi);
void SH_project_polar_function(SH_polar_fn fn, const SHSample samples[], double result[])
{
	const double weight = 4.0*PI;
	// for each sample
	for(int i=0; i<n_samples; ++i) {
		double theta = samples[i].sph.x;
		double phi = samples[i].sph.y;
		for(int n=0; n<n_coeff; ++n) {
			result[n] += fn(theta,phi) * samples[i].coeff[n];
		}
	}
		// divide the result by weight and number of samples
	double factor = weight / n_samples;
	for(i=0; i<n_coeff; ++i) {
		result[i] = result[i] * factor;
	}
}
