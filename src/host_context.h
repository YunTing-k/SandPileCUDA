//+FHDR//////////////////////////////////////////////////////////////////////////////
// Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab
// Author: Yu Huang
// Coding: UTF-8
// Create Date: 2025.10.08
// Description: 
// SandPile context definitions in the host side during simulation and postprocess.
//
// Revision:
// ---------------------------------------------------------------------------------
// [Date]         [By]         [Version]         [Change Log]
// ---------------------------------------------------------------------------------
// 2025/10/08     Yu Huang     1.0               First implementation
// ---------------------------------------------------------------------------------
//
//-FHDR//////////////////////////////////////////////////////////////////////////////
#ifndef HOST_CONTEXT_H
#define HOST_CONTEXT_H

#include <Eigen/Eigen>
using Eigen::RowMajor;
using Eigen::ColMajor;

/* Context of SandPile simulator in host side */
typedef struct SandPileContextHost {
    int *pile_host;                       // pile matrix in host side (row-major)
    unsigned long long int count;         // total collapse count in host side
    int lut_r[6], lut_g[6], lut_b[6];     // color lookup table in host side
    Eigen::MatrixXi r_mat, g_mat, b_mat;  // frame's R/G/B matrix in host side (col-major)
} SandPileContextHost;

#endif
