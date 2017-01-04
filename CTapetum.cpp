#include "CTapetum.h"
#include "ADMMDenoise.h"
#include <armadillo>

using namespace arma;

void denoise(double const * const image, double * const result, int ny, int nx, double smoothing_parameter){

    mat src_image(image,nx,ny);
    mat dest_image(nx,ny);

    admmdenoise(src_image,dest_image,smoothing_parameter);

    memcpy(result,(double*)dest_image.memptr(),(ny*nx)*sizeof(double));
}
