//+FHDR//////////////////////////////////////////////////////////////////////////////
// Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab
// Author: Yu Huang
// Coding: UTF-8
// Create Date: 2025.10.08
// Description: 
// SandPile context definitions in the cuda side during simulation and postprocess.
//
// Revision:
// ---------------------------------------------------------------------------------
// [Date]         [By]         [Version]         [Change Log]
// ---------------------------------------------------------------------------------
// 2025/10/08     Yu Huang     1.0               First implementation
// ---------------------------------------------------------------------------------
//
//-FHDR//////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H

/* Context of SandPile simulator in host side */
typedef struct SandPileContextDevice {
    int *pile_device;                     // pile matrix in device side (row-major)
    int *pile_diff1, *pile_diff2;         // ping-pang buffer of pile diff matrix in device side
    unsigned long long int *count;        // total collapse count in device side
    int *lut_r, *lut_g, *lut_b;           // color lookup table in device side
    int *r_mat, *g_mat, *b_mat;           // frame's R/G/B matrix in device side (col-major)
} SandPileContextDevice;

#endif
