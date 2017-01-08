#ifndef CONVEXSEGMENT_H_
#define CONVEXSEGMENT_H_

#include <armadillo>



void GCSegment_1c(arma::mat & image, arma::mat & output_segments, double channel_averages[2]);
void GCS_threshold(arma::mat & raw_segmentation, double threshold);
void GCS_segments_to_image(arma::mat & segments, arma::mat & image_result, double channel_averages[2]);
#endif
