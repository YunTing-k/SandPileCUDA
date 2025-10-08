//+FHDR//////////////////////////////////////////////////////////////////////////////
// Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab
// Author: Yu Huang
// Coding: UTF-8
// Create Date: 2025.3.21
// Description: 
// Sandpile fractal simulator
//
// Revision:
// ---------------------------------------------------------------------------------
// [Date]         [By]         [Version]         [Change Log]
// ---------------------------------------------------------------------------------
// 2025/03/21     Yu Huang     1.0               First implementation
// 2025/10/08     Yu Huang     1.1               Merge the pile iteration function
// 2025/10/08     Yu Huang     1.2               Video output realization
// ---------------------------------------------------------------------------------
//
//-FHDR//////////////////////////////////////////////////////////////////////////////
#include "pile.h"

/**
 * [CUDA] Sandpile simulation iteration.
 * 
 * 
 * @param p pointer of the pile param struct
 * @param cur_pile current pile matrix
 * @param diff_in input pile diff matrix
 * @param diff_out output pile diff matrix
 * @param count total collapse count
 * @param sys_log logger of the simulator
 * @return NULL
 * 
 * 
 */
void pile_itr(const PileParam *p, int *cur_pile, int *diff_in, int *diff_out, unsigned long long int *count, const logger &sys_log) {
    switch (p->shape) {
        case TRIANGLE:
            call_pile_itr_tri(p, cur_pile, diff_in, diff_out, count);
            break;
        case QUADRILATERAL:
            call_pile_itr_quad(p, cur_pile, diff_in, diff_out, count);
            break;
        case HEXAGON:
            call_pile_itr_hex(p, cur_pile, diff_in, diff_out, count);
            break;
        default:
            SPDLOG_LOGGER_ERROR(sys_log, "Invalid type shape: {}", (int)p->shape);
            throw std::runtime_error("Invalid type shape");
    }
}

/**
 * [CPU] Get the visualization results according to lut.
 * 
 * 
 * @param pile_p pointer of the pile param struct
 * @param ctx_host TECoSim context in the host side
 * @return NULL
 * 
 * 
 */
void visualize_host(const PileParam *pile_p, SandPileContextHost &ctx_host) {
    const int width = pile_p->width;
    const int height = pile_p->height;

    #pragma omp parallel for collapse(2)
     for (int j = 0; j < width; j++) { // col
        for (int i = 0; i < height; i++) { // row
            int idx = ctx_host.pile_host[i * width + j];
            if (idx < 6) {
                ctx_host.r_mat(i, j) = ctx_host.lut_r[idx];
                ctx_host.g_mat(i, j) = ctx_host.lut_g[idx];
                ctx_host.b_mat(i, j) = ctx_host.lut_b[idx];
            } else {
                ctx_host.r_mat(i, j) = 0;
                ctx_host.g_mat(i, j) = 0;
                ctx_host.b_mat(i, j) = 0;
            }
        }
    }
}
