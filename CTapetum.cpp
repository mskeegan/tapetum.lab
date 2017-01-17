#include "CTapetum.h"
#include "ADMMDenoise.h"
#include "ConvexSegment.h"
#include <armadillo>

using namespace arma;
using namespace std;

void denoise(double const * const image, double * const result, int ny, int nx, double smoothing_parameter){

    mat src_image(image,nx,ny);
    mat dest_image(nx,ny,fill::zeros);

    admmdenoise(src_image,dest_image,smoothing_parameter);

    memcpy(result,(double*)dest_image.memptr(),(ny*nx)*sizeof(double));
}

void segment(const double* const image, double* const result ,int ny, int nx, double channels[2]){
    
    mat src_image(image,nx,ny);
    mat dest_image(nx,ny,fill::zeros);

    GCSegment_1c(src_image,dest_image,channels);
    memcpy(result,(double*)dest_image.memptr(),(ny*nx)*sizeof(double));
}

void segmentmp(const double* const image, double* const result ,int ny, int nx, int nc, double* channels){
    mat src_image(image,nx,ny);
    mat segmatrix(nx,ny,fill::zeros);
    vector<mat> raw_segments(nc,segmatrix);
    double smooth_weight = 0.1;

    GCSegment_SplitBregman_Multiphase(src_image, raw_segments, nc, channels, smooth_weight);

    for (int i = 0; i < nc; i++){
	memcpy(result+i*nx*ny, (double*)raw_segments[i].memptr(),(ny*nx)*sizeof(double));
    }
}

void discrete_image(const double * const segments, double * const image, int ny, int nx, int nc, double * const channels){

    if(nc == 2){
	mat src_image(segments,nx,ny);
	mat dest_image(nx,ny,fill::zeros);

	GCS_segments_to_image(src_image,dest_image,channels);
	memcpy(image,(double*)dest_image.memptr(),(ny*nx)*sizeof(double));
    }
    else if (nc > 2){
	mat dest_image(nx,ny,fill::zeros);
	vector<mat> src_segment;
	for (int i = 0; i < nc; i++){
	    mat src(segments+i*nx*ny,nx,ny);
	    src_segment.push_back(src);
	}

	GCS_segments_to_image(src_segment,dest_image,channels);
	memcpy(image,(double*)dest_image.memptr(),(ny*nx)*sizeof(double));
    }
}
