#ifndef CONVEXSEGMENT_H_
#define CONVEXSEGMENT_H_

#include <armadillo>
#include <vector>

void GCSegment_1c(arma::mat & image, arma::mat & output_segments, double channel_averages[2]);
void GCSegment_SplitBregman_Multiphase(const arma::mat & image, std::vector<arma::mat> & segments, int num_channels, double * channel_averages, double lambda);

void GCS_threshold(const std::vector<arma::mat> & raw_segments, arma::umat & labels);

void GCS_threshold(arma::mat & raw_segmentation, double threshold);
void GCS_segments_to_image(arma::mat & segments, arma::mat & image_result, double channel_averages[2]);
//void GCS_segments_to_image(std::vector<arma::mat> & segments, arma::mat & image_result, double* channel_averages){}

void Simplex_projection(arma::vec & c);

#endif
