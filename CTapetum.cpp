#include "CTapetum.h"
#include "ADMMDenoise.h"
#include "ConvexSegment.h"
#include <armadillo>

using namespace arma;
using namespace std;

void denoise(double const * const image, double * const result, int ny, int nx, double smoothing_parameter){

    const mat src_image(image,ny,nx);
    mat dest_image(result,ny,nx,false,false);

    admmdenoise(src_image,dest_image,smoothing_parameter);
}

void segment(const double* const image, double* const result ,int ny, int nx, double channels[2]){
    
    const mat src_image(image,ny,nx);
    mat dest_image(result,ny,nx,false,false);

    GCSegment_1c(src_image,dest_image,channels);
    //memcpy(result,(double*)dest_image.memptr(),(ny*nx)*sizeof(double));
}

void segmentmp(const double* const image, double* const result ,int ny, int nx, int nc, double* channels){
    const mat src_image(image,ny,nx);
    cube output_array(result,ny,nx,nc,false,false);
    
    mat segmatrix(ny,nx,fill::zeros);
    vector<mat> raw_segments(nc,segmatrix);
    vector<mat> next_version_segments(nc);
    for (int i = 0; i < nc; i++){
	mat segment_slice(result+i*nx*ny,ny,nx);
	next_version_segments.push_back(segment_slice);
    }
    double smooth_weight = 0.1;

    GCSegment_SplitBregman_Multiphase(src_image, raw_segments, nc, channels, smooth_weight);

    for (int i = 0; i < nc; i++){
	memcpy(result+i*nx*ny, (double*)raw_segments[i].memptr(),(ny*nx)*sizeof(double));
    }
}

void discrete_image(const double * const segments, double * const image, int ny, int nx, int nc, double * const channels){

    mat dest_image(image,ny,nx,false,false);
    if(nc == 2){
	const mat src_segments(segments,ny,nx);
	GCS_segments_to_image(src_segments,dest_image,channels);
    }
    else if (nc > 2){
	//mat dest_image(ny,nx,fill::zeros);
	vector<mat> src_segment;
	for (int i = 0; i < nc; i++){
	    mat src(segments+i*nx*ny,ny,nx);
	    src_segment.push_back(src);
	}

	GCS_segments_to_image(src_segment,dest_image,channels);
	//memcpy(image,(double*)dest_image.memptr(),(ny*nx)*sizeof(double));
    }
}
