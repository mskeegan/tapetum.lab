#ifndef ADMMDENOISE_H_
#define ADMMDENOISE_H_

#include <armadillo>

using namespace arma;

void admmdenoise(const mat & image, mat & solution, const double smooth_weight=0.1*255);

#endif /*ADMMDENOISE_H_*/
