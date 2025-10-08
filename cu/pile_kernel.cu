//+FHDR//////////////////////////////////////////////////////////////////////////////
// Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab
// Author: Yu Huang
// Coding: UTF-8
// Create Date: 2025.3.21
// Description: 
// Sandpile fractal simulator cuda realization
//
// Revision:
// ---------------------------------------------------------------------------------
// [Date]         [By]         [Version]         [Change Log]
// ---------------------------------------------------------------------------------
// 2025/03/21     Yu Huang     1.0               First implementation
// 2025/04/09     Yu Huang     1.1               Add collapse counter
// ---------------------------------------------------------------------------------
//
//-FHDR//////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include "../src/pile_param.h"

/**
 * Initialize the pile
 * 
 * 
 * @param idx index of initialized pile
 * @param sand_num sand num of initialized pile
 * @param cur_pile pile data for initialization
 * @param count count of the sandpile collapse
 * @return NULL
 * 
 * 
 */
__global__ void pile_initialize(const int idx, const int sand_num, int *cur_pile) {
    if (threadIdx.x == 0) {
        cur_pile[idx] = sand_num;
    }
}

/**
 * Caller of the pile_initialize
 * 
 * 
 * @param p pointer of the param struct
 * @param cur_pile pile data for initialization
 * @param count count of the sandpile collapse
 * @return NULL
 * 
 * 
 */
extern "C" void call_pile_initialize(const PileParam *p, int *cur_pile) {
    int idx = p->width * (p->height / 2 - 1) + p->width / 2 - 1;
    pile_initialize<<<1, 1>>>(idx, p->ini_sand_num, cur_pile);
}

/**
 * pile iterator (TRIANGLE)
 * 
 * 
 * @param cur_pile pile data for initialization
 * @param diff_in input pile diff
 * @param diff_out output pile diff
 * @param width width of sandbox
 * @param height height of sandbox
 * @return NULL
 * 
 * 
 */
__global__ void pile_itr_tri(int *cur_pile, int *diff_in, int *diff_out, int width, int height, unsigned long long int *count) {
    // index of threads
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    cur_pile[row * width + col] += diff_in[row * width + col];

    if (cur_pile[row * width + col] >= 3) {
        atomicAdd(&diff_out[row * width + col], -3);
        atomicAdd(count, 1);
        if ((row * width + col) % 2 == 0) { // up triangle
            if (row != (height - 1)) { // not the bottom edge
                atomicAdd(&diff_out[(row + 1) * width + col], 1);
            }
            if (col != 0) { // not the left edge
                atomicAdd(&diff_out[row * width + (col - 1)], 1);
            }
            if (col != (width - 1)) { // not the right edge
                atomicAdd(&diff_out[row * width + (col + 1)], 1);
            }
        } else { // down triangle
            if (row != 0) { // not the top edge
                atomicAdd(&diff_out[(row - 1) * width + col], 1);
            }
            if (col != 0) { // not the left edge
                atomicAdd(&diff_out[row * width + (col - 1)], 1);
            }
            if (col != (width - 1)) { // not the right edge
                atomicAdd(&diff_out[row * width + (col + 1)], 1);
            }
        }
    }
}

/**
 * Caller of the pile iterator (TRIANGLE)
 * 
 * 
 * @param p pointer of the param struct
 * @param cur_pile pile data for initialization
 * @param diff_in input pile diff
 * @param diff_out output pile diff
 * @return NULL
 * 
 * 
 */
extern "C" void call_pile_itr_tri(const PileParam *p, int *cur_pile, int *diff_in, int *diff_out, unsigned long long int *count) {

    int bk_row, bk_col;
    bk_col = p->width / 8;
    bk_row = p->height / 8;
    dim3 num_block(bk_col, bk_row);
    dim3 threads_block(8, 8); // x(col) = 8 threads, y(row) = 8 threads

    pile_itr_tri<<<num_block, threads_block>>>(cur_pile, diff_in, diff_out, p->width, p->height, count);
}

/**
 * pile iterator (QUADRILATERAL)
 * 
 * 
 * @param cur_pile pile data for initialization
 * @param diff_in input pile diff
 * @param diff_out output pile diff
 * @param width width of sandbox
 * @param height height of sandbox
 * @return NULL
 * 
 * 
 */
__global__ void pile_itr_quad(int *cur_pile, int *diff_in, int *diff_out, int width, int height, unsigned long long int *count) {
    // index of threads
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    cur_pile[row * width + col] += diff_in[row * width + col];

    if (cur_pile[row * width + col] >= 4) {
        atomicAdd(&diff_out[row * width + col], -4);
        atomicAdd(count, 1);
        if (row != 0) { // not the top edge
            atomicAdd(&diff_out[(row - 1) * width + col], 1);
        }
        if (row != (height - 1)) { // not the bottom edge
            atomicAdd(&diff_out[(row + 1) * width + col], 1);
        }
        if (col != 0) { // not the left edge
            atomicAdd(&diff_out[row * width + (col - 1)], 1);
        }
        if (col != (width - 1)) { // not the right edge
            atomicAdd(&diff_out[row * width + (col + 1)], 1);
        }
    }
}

/**
 * Caller of the pile iterator (QUADRILATERAL)
 * 
 * 
 * @param p pointer of the param struct
 * @param cur_pile pile data for initialization
 * @param diff_in input pile diff
 * @param diff_out output pile diff
 * @return NULL
 * 
 * 
 */
extern "C" void call_pile_itr_quad(const PileParam *p, int *cur_pile, int *diff_in, int *diff_out, unsigned long long int *count) {

    int bk_row, bk_col;
    bk_col = p->width / 8;
    bk_row = p->height / 8;
    dim3 num_block(bk_col, bk_row);
    dim3 threads_block(8, 8); // x(col) = 8 threads, y(row) = 8 threads
    
    pile_itr_quad<<<num_block, threads_block>>>(cur_pile, diff_in, diff_out, p->width, p->height, count);
}

/**
 * pile iterator (HEXAGON)
 * 
 * 
 * @param cur_pile pile data for initialization
 * @param diff_in input pile diff
 * @param diff_out output pile diff
 * @param width width of sandbox
 * @param height height of sandbox
 * @return NULL
 * 
 * 
 */
__global__ void pile_itr_hex(int *cur_pile, int *diff_in, int *diff_out, int width, int height, unsigned long long int *count) {
    // index of threads
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    cur_pile[row * width + col] += diff_in[row * width + col];

    if (cur_pile[row * width + col] >= 6) {
        atomicAdd(&diff_out[row * width + col], -6);
        atomicAdd(count, 1);
        if (col % 2 == 0) { // up hexagon
            if (row != 0) { // not the top edge
                atomicAdd(&diff_out[(row - 1) * width + col], 1);
            }
            if (row != (height - 1)) { // not the bottom edge
                atomicAdd(&diff_out[(row + 1) * width + col], 1);
            }
            if (col != 0) { // not the left edge
                atomicAdd(&diff_out[row * width + (col - 1)], 1); // left-down
                if (row != 0) { // left-up
                    atomicAdd(&diff_out[(row - 1) * width + (col - 1)], 1);
                }
            }
            if (col != (width - 1)) { // not the right edge
                atomicAdd(&diff_out[row * width + (col + 1)], 1); // right-down
                if (row != 0) { // right-up
                    atomicAdd(&diff_out[(row - 1) * width + (col + 1)], 1);
                }
            }
        } else { // down hexagon
            if (row != 0) { // not the top edge
                atomicAdd(&diff_out[(row - 1) * width + col], 1);
            }
            if (row != (height - 1)) { // not the bottom edge
                atomicAdd(&diff_out[(row + 1) * width + col], 1);
            }
            // it can't be left edge
            atomicAdd(&diff_out[row * width + (col - 1)], 1); // left-up
            if (row != (height - 1)) { // left-down
                atomicAdd(&diff_out[(row + 1) * width + (col - 1)], 1);
            }
            if (col != (width - 1)) { // not the right edge
                atomicAdd(&diff_out[row * width + (col + 1)], 1); // right-up
                if (row != (height - 1)) { // right-down
                    atomicAdd(&diff_out[(row + 1) * width + (col + 1)], 1);
                }
            }
        }
    }
}

/**
 * Caller of the pile iterator (HEXAGON)
 * 
 * 
 * @param p pointer of the param struct
 * @param cur_pile pile data for initialization
 * @param diff_in input pile diff
 * @param diff_out output pile diff
 * @return NULL
 * 
 * 
 */
extern "C" void call_pile_itr_hex(const PileParam *p, int *cur_pile, int *diff_in, int *diff_out, unsigned long long int *count) {

    int bk_row, bk_col;
    bk_col = p->width / 8;
    bk_row = p->height / 8;
    dim3 num_block(bk_col, bk_row);
    dim3 threads_block(8, 8); // x(col) = 8 threads, y(row) = 8 threads
    
    pile_itr_hex<<<num_block, threads_block>>>(cur_pile, diff_in, diff_out, p->width, p->height, count);
}
