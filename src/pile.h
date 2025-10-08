#ifndef pile_H
#define pile_H

#include <iostream>
#include <iomanip>
#include <omp.h>
#include <cuda_runtime.h>
#include "utility.h"
#include "host_context.h"
#include "cuda_context.h"

void pile_itr(const PileParam *p, int *cur_pile, int *diff_in, int *diff_out, unsigned long long int *count, const logger &sys_log);
void visualize_host(const PileParam *pile_p, SandPileContextHost &host_ctx);

extern "C"
{
    void call_pile_initialize(const PileParam *p, int *cur_pile);
    void call_pile_itr_tri(const PileParam *p, int *cur_pile, int *diff_in, int *diff_out, unsigned long long int *count);
    void call_pile_itr_quad(const PileParam *p, int *cur_pile, int *diff_in, int *diff_out, unsigned long long int *count);
    void call_pile_itr_hex(const PileParam *p, int *cur_pile, int *diff_in, int *diff_out, unsigned long long int *count);
    void call_visualize_cuda(const PileParam *p, SandPileContextDevice &ctx_device);
}

#endif
