#ifndef CTAPETUM_H_
#define CTAPETUM_H_

extern "C"{
    void denoise(double const * const image, double * const result, int ny, int nx, double smoothing_parameter);

    void segment_1c(const double * const image, double* const segments, int ny, int nx, double channels[2]);
    void discrete_image(const double * const segments, double* const image, int ny, int nx, double channels[2]);
}

#endif
