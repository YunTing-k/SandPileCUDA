#ifndef pile_H
#define pile_H

#include <iostream>
#include <iomanip>
#include <omp.h>
#include <cuda_runtime.h>
#include "utility.h"

extern "C"
{
    void call_pile_initialize(const PARAM *p, int *cur_pile);
    void call_pile_itr_tri(const PARAM *p, int *cur_pile, int *diff_in, int *diff_out, unsigned long long int *count);
    void call_pile_itr_quad(const PARAM *p, int *cur_pile, int *diff_in, int *diff_out, unsigned long long int *count);
    void call_pile_itr_hex(const PARAM *p, int *cur_pile, int *diff_in, int *diff_out, unsigned long long int *count);
}

#endif
