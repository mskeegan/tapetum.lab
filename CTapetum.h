#ifndef CTAPETUM_H_
#define CTAPETUM_H_

extern "C"{
    void denoise(const double* const image, double * const result, int ny, int nx, double smoothing_parameter);
    
    void segment(const double * const image, double* const result, int ny, int nx, double channels[2]);
    void segmentmp(const double* const image, double* const result ,int ny, int nx, int nc, double* channels);
    void getimage(const double * const segments, double * const image, int ny, int nx, int nc, double * const channels);
}

#endif
