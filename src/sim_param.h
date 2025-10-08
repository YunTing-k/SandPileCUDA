//+FHDR//////////////////////////////////////////////////////////////////////////////
// Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab
// Author: Yu Huang
// Coding: UTF-8
// Create Date: 2025.10.08
// Description: 
// Sandpile fractal simulation param definitions
//
// Revision:
// ---------------------------------------------------------------------------------
// [Date]         [By]         [Version]         [Change Log]
// ---------------------------------------------------------------------------------
// 2025/10/08     Yu Huang     1.0               Add sim params
// ---------------------------------------------------------------------------------
//
//-FHDR//////////////////////////////////////////////////////////////////////////////
#ifndef SIMPARAM_H
#define SIMPARAM_H
#include <string>

/* frame sequence format */
enum frame_sequence_format {
    SEQUENCE_JPEG,                       // JPEG format
    SEQUENCE_PNG,                        // PNG format
    SEQUENCE_TIFF,                       // TIFF format
    SEQUENCE_BMP,                        // BMP format
    SEQUENCE_JPEG2000                    // JPEG2000 format
};

/* Struct of parameters */
typedef struct SimParam {
    int max_itr_steps;                   // max iteration steps
    std::string data_path;               // path of output raw data
    std::string video_path;              // path of output video
    frame_sequence_format outseq_format; // output frame sequence format
    int sp_rate;                         // sample rate for frame sequence
    /* video codec params */
    long long bit_rate;                  // output visualization bit rate (bit/s)
    long long rc_max_rate;               // maximum bitrate (bit/s)
    long long rc_min_rate;               // minimum bitrate (bit/s)
    int rc_buffer_size;                  // decoder bitstream buffer size (bits)
    int gop_size;                        // key frame amount
    int max_b_frames;                    // max bidirectional frame amount
    int thread_count;                    // encoding thread amount
} SimParam;
#endif
