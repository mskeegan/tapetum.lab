#ifndef ADMMDENOISE_H_
#define ADMMDENOISE_H_

#include <armadillo>

using namespace arma;

void admmdenoise(const mat & image, mat & solution, const double smooth_weight=0.1*255);
void ad_core_iteration(const mat & image, mat & u, mat & bx, mat & by, mat & dx, mat & dy, double mu);

#endif /*ADMMDENOISE_H_*/
