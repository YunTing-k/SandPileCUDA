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
// ---------------------------------------------------------------------------------
//
//-FHDR//////////////////////////////////////////////////////////////////////////////
#include "pile.h"

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
