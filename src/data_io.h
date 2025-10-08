#ifndef DATA_IO_H
#define DATA_IO_H

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include <libavutil/imgutils.h>
    #include <libavutil/opt.h>
    #include <libavutil/error.h>
    #include <libavfilter/buffersrc.h>
    #include <libavfilter/buffersink.h>
}
#include <iostream>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <omp.h>
#include "utility.h"

AVFormatContext* open_video(const char* file_name, const logger &sys_log);
int find_video_stream_idx(const AVFormatContext* video_format_ctx, const logger &sys_log);
const AVCodec* get_codec_from_stream(const AVFormatContext* in_format_ctx, int in_stream_idx, const logger &sys_log);
const AVCodec* get_codec_from_id(AVCodecID in_codec_id, const logger &sys_log);
AVCodecContext* prepare_codec(const AVCodec* in_codec, AVCodecParameters* in_codec_params, const logger &sys_log);
AVFormatContext* get_outcontext_by_name(const char* format_name, const logger &sys_log);
void save_frame2ppm(AVFrame* frame, const char* file_name, const logger &sys_log);
void save_frame2bin(AVFrame* frame, const char* file_name, const logger &sys_log);
void save_int2bin(const PileParam *p, const int *int_mat, const char* file_name, const logger &sys_log);
#endif
