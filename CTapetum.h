#ifndef CTAPETUM_H_
#define CTAPETUM_H_

extern "C"{
    void denoise(const double* image, double* result, int ny, int nx, double smoothing_parameter);
}

#endif
