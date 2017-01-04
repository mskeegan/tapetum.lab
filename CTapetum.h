#ifndef CTAPETUM_H_
#define CTAPETUM_H_

extern "C"{
    void denoise(double const * const image, double * const result, int ny, int nx, double smoothing_parameter);
}

#endif
