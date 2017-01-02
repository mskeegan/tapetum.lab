#include "ADMMDenoise.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>

using namespace arma;

void admmdenoise(const mat & image, mat & solution, const double smooth_weight){

	// ***	The l2-penalty parameter (Lagrange multiplier)
	const double lambda = smooth_weight/2.0;
	// *** Parameter mu - data-fidelity parameter
	const double mu = smooth_weight;
	
	int row = image.n_rows;
	int col = image.n_cols;
	
	mat u(image);
	mat bx(row, col, fill::zeros);
	mat by(row, col, fill::zeros);
	mat dx(row, col, fill::zeros);
	mat dy(row, col, fill::zeros);
	mat temp(row, col, fill::zeros);
	
	double tol = 1e-5;
	double rel_error = 1.0;

	double ux, uy, s;
	double max_new = 0, max_old;
	int iterations = 0;

	// *** Split Bregman Algorithm for denoising *** //
	while (rel_error > tol){
		rel_error = 0.0;
		max_old = max_new;
		max_new = 0;

		for(int y = 1; y < col-1; y++){
		        for(int x = 1; x < row-1; x++){
				ux = u(x+1,y) - u(x,y);
				uy = u(x,y+1) - u(x,y);
				s = sqrt( pow(ux+bx(x,y),2) + pow(uy+by(x,y),2) );
				dx(x,y) = (s == 0) ? 0.0 : ((ux + bx(x,y)) / s) * fmax(s-1/lambda, 0.0);
				dy(x,y) = (s == 0) ? 0.0 : ((uy + by(x,y)) / s) * fmax(s-1/lambda, 0.0);
				bx(x,y) += ux - dx(x,y);
				by(x,y) += uy - dy(x,y);
			}
		}

		for(int y = 1; y < col - 1; y++){
		        for(int x = 1; x < row - 1; x++){
				temp(x,y) = -(dx(x,y) - dx(x-1,y) + dy(x,y) - dy(x,y-1));
				temp(x,y) += (bx(x,y) - bx(x-1,y) + by(x,y) - by(x,y-1));
				temp(x,y) = lambda*temp(x, y) + mu*image(x,y) + lambda*(u(x-1,y) + u(x+1,y) + u(x,y-1) + u(x,y+1));
				temp(x,y) = temp(x,y)/(mu+4*lambda);
				rel_error += pow(u(x,y) - temp(x,y),2);
				u(x,y) = temp(x,y);
				max_new += u(x,y);
			}

		u(row-1,y) = u(row-2,y);
		}

		for(int x = 1; x < row - 1; x++){
		        u(x, col-1) = u(x, col-2);
		}

		rel_error = rel_error/fmax(max_new, max_old);

		iterations++;
	}

	cout << "Split Bregman Denoise, Iterations: " << iterations << ", Error Tolerance: " << tol << endl; 
	solution = u;
};
