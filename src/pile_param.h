//+FHDR//////////////////////////////////////////////////////////////////////////////
// Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab
// Author: Yu Huang
// Coding: UTF-8
// Create Date: 2025.3.21
// Description: 
// Sandpile fractal simulation param definitions
//
// Revision:
// ---------------------------------------------------------------------------------
// [Date]         [By]         [Version]         [Change Log]
// ---------------------------------------------------------------------------------
// 2025/03/21     Yu Huang     1.0               First implementation
// ---------------------------------------------------------------------------------
//
//-FHDR//////////////////////////////////////////////////////////////////////////////
#ifndef PILEPARAM_H
#define PILEPARAM_H

/* Pile grid shape */
enum grid_shape {
    TRIANGLE = 1,         // triangle 
    QUADRILATERAL = 2,    // quadrilaterals
    HEXAGON = 3           // hexagon
};

/* Struct of parameters */
typedef struct PileParam {
    grid_shape shape;     // grid shape

    int width;            // sandbox width
    int height;           // sandbox height
    int ini_sand_num;     // initial sand num on center cell
} PileParam;
#endif
