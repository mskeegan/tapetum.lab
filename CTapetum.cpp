#include "CTapetum.h"
#include "ADMMDenoise.h"
#include "ConvexSegment.h"
#include <armadillo>

using namespace arma;

void denoise(double const * const image, double * const result, int ny, int nx, double smoothing_parameter){

    mat src_image(image,nx,ny);
    mat dest_image(nx,ny);

    admmdenoise(src_image,dest_image,smoothing_parameter);

    memcpy(result,(double*)dest_image.memptr(),(ny*nx)*sizeof(double));
}

void segment_1c(const double* const image, double* const segments ,int ny, int nx, double channels[2]){
    
    mat src_image(image,nx,ny);
    mat dest_image(nx,ny,fill::zeros);

    GCSegment_1c(src_image,dest_image,channels);
    memcpy(segments,(double*)dest_image.memptr(),(ny*nx)*sizeof(double));
}

void discrete_image(const double * const segments, double* const image, int ny, int nx, double channels[2]){

    mat src_image(segments,nx,ny);
    mat dest_image(nx,ny,fill::zeros);

    GCS_segments_to_image(src_image,dest_image,channels);
    memcpy(image,(double*)dest_image.memptr(),(ny*nx)*sizeof(double));
}
