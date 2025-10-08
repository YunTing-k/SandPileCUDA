//+FHDR//////////////////////////////////////////////////////////////////////////////
// Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab
// Author: Yu Huang
// Coding: UTF-8
// Create Date: 2025.2.28
// Description: 
// Functions definition of data io management of Video, Images and Binary files for
// simulation and postprocess.
//
// Revision:
// ---------------------------------------------------------------------------------
// [Date]         [By]         [Version]         [Change Log]
// ---------------------------------------------------------------------------------
// 2025/02/28     Yu Huang     1.0               First implementation
// 2025/03/04     Yu Huang     1.1               Output IO realization
// 2025/10/08     Yu Huang     1.2               Remove namespace & Adjust logger
// ---------------------------------------------------------------------------------
//
//-FHDR//////////////////////////////////////////////////////////////////////////////
#include "data_io.h"

/**
 * open the video by pathname/URL.
 * 
 * 
 * @param file_name Pathname or URL of the video
 * @param sys_log logger of the simulator
 * @return AVFormatContext of the opened video
 * 
 * 
 */
AVFormatContext* open_video(const char* file_name, const logger &sys_log) {
    AVFormatContext* video_format_ctx = nullptr;
    if (avformat_open_input(&video_format_ctx, file_name, nullptr, nullptr) != 0) {
        SPDLOG_LOGGER_ERROR(sys_log, "Open video failed!");
        throw std::runtime_error("Could not open video file");
    }
    if (avformat_find_stream_info(video_format_ctx, nullptr) < 0) {
        SPDLOG_LOGGER_ERROR(sys_log, "Find video stream failed!");
        throw std::runtime_error("Could not find stream information");
    }
    SPDLOG_LOGGER_INFO(sys_log, "Input Stream Information:");
    SPDLOG_LOGGER_INFO(sys_log, "-- Number of Streams: {}", video_format_ctx->nb_streams);
    SPDLOG_LOGGER_INFO(sys_log, "-- Start Time: {:.3f} s", (double)(video_format_ctx->start_time / AV_TIME_BASE));
    SPDLOG_LOGGER_INFO(sys_log, "-- Duration Time: {:.3f} s", (double)(video_format_ctx->duration / AV_TIME_BASE));
    SPDLOG_LOGGER_INFO(sys_log, "-- Bit Rate: {} bit/s", video_format_ctx->bit_rate);
    return video_format_ctx;
}

/**
 * find the video stream index from AVFormatContext.
 * 
 * 
 * @param video_format_ctx target AVFormatContext
 * @param sys_log logger of the simulator
 * @return index of the video stream in input AVFormatContext
 * 
 * 
 */
int find_video_stream_idx(const AVFormatContext* video_format_ctx, const logger &sys_log) {
    int video_stream_idx = -1;
    for (unsigned int i = 0; i < video_format_ctx->nb_streams; i++) {
        // AVStream -> AVCodecParameters -> AVMediaType to find video data stream
        if (video_format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            break;
        }
    }
    if (video_stream_idx == -1) {
        SPDLOG_LOGGER_ERROR(sys_log, "Find video stream index failed!");
        throw std::runtime_error("Could not find video stream");
    }
    SPDLOG_LOGGER_INFO(sys_log, "-- Video Stream Index: {}", video_stream_idx);
    return video_stream_idx;
}

/**
 * get codec from the stream.
 * 
 * 
 * @param in_format_ctx target AVFormatContext
 * @param in_stream_idx target stream index
 * @param sys_log logger of the simulator
 * @return AVCodec from the input stream index
 * 
 * 
 */
const AVCodec* get_codec_from_stream(const AVFormatContext* in_format_ctx, int in_stream_idx, const logger &sys_log) {
    const AVCodecParameters* codec_params = in_format_ctx->streams[in_stream_idx]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
    if (!codec) {
        SPDLOG_LOGGER_ERROR(sys_log, "Unsupported codec");
        throw std::runtime_error("Could not get codec");
    }
    return codec;
}

/**
 * get codec from the id.
 * 
 * 
 * @param in_codec_id target codec id
 * @param sys_log logger of the simulator
 * @return AVCodec according to the codec id
 * 
 * 
 */
const AVCodec* get_codec_from_id(AVCodecID in_codec_id, const logger &sys_log) {
    const AVCodec* codec = avcodec_find_encoder(in_codec_id);
    if (!codec) {
        SPDLOG_LOGGER_ERROR(sys_log, "Unsupported codec");
        throw std::runtime_error("Could not get codec");
    }
    return codec;
}

/**
 * allocate codec context and open the codec.
 * 
 * 
 * @param in_codec input codec
 * @param in_codec_params input codec parameters
 * @param sys_log logger of the simulator
 * @return AVCodecContext of the target codec
 * 
 * 
 */
AVCodecContext* prepare_codec(const AVCodec* in_codec, AVCodecParameters* in_codec_params, const logger &sys_log) {
    AVCodecContext* codec_ctx = avcodec_alloc_context3(in_codec);
    avcodec_parameters_to_context(codec_ctx, in_codec_params);
    if (avcodec_open2(codec_ctx, in_codec, nullptr) < 0) {
        SPDLOG_LOGGER_ERROR(sys_log, "Codec open failed!");
        throw std::runtime_error("Could not open codec");
    }
    return codec_ctx;
}

/**
 * get the output context by filename.
 * 
 * 
 * @param format_name format name that you wish to save the output contents, e.g., xx.mp4
 * @param sys_log logger of the simulator
 * @return output AVFormatContext by filename.
 * 
 * 
 */
AVFormatContext* get_outcontext_by_name(const char* format_name, const logger &sys_log) {
    AVFormatContext* out_format_ctx = nullptr;
    avformat_alloc_output_context2(&out_format_ctx, nullptr, nullptr, format_name);
    if (!out_format_ctx) {
        SPDLOG_LOGGER_ERROR(sys_log, "Output context allocate failed!");
        throw std::runtime_error("Could not create output context");
    }
    return out_format_ctx;
}

/**
 * save RGB24 frame to file in PPM files.
 * 
 * 
 * @param frame input RGB24 format frame
 * @param file_name output filename
 * @param sys_log logger of the simulator
 * @return NULL
 * 
 * 
 */
void save_frame2ppm(AVFrame* frame, const char* file_name, const logger &sys_log) {
    FILE* file = fopen(file_name, "wb");
    if (!file) {
        SPDLOG_LOGGER_ERROR(sys_log, "Could not open file: {}", file_name);
        fclose(file);
        return;
    }
    // PPM header
    fprintf(file, "P6\n%d %d\n255\n", frame->width, frame->height);
    for (int row = 0; row < frame->height; row++) {
        fwrite(frame->data[0] + row * frame->linesize[0], 1, frame->width * 3, file);
    }
    fclose(file);
}

/**
 * save frame to file in BIN.
 * 
 * 
 * @param frame input RGB24 format frame
 * @param file_name output filename
 * @param sys_log logger of the simulator
 * @return NULL
 * 
 * 
 */
void save_frame2bin(AVFrame* frame, const char* file_name, const logger &sys_log) {
    FILE* file = fopen(file_name, "wb");
    if (!file) {
        SPDLOG_LOGGER_ERROR(sys_log, "Could not open file: {}", file_name);
        fclose(file);
        return;
    }
    for (int row = 0; row < frame->height; row++) {
        fwrite(frame->data[0] + row * frame->linesize[0], 1, frame->width * 3, file);
    }
    fclose(file);
}

/**
 * save int eigen matrix/vector into binary file.
 * 
 * 
 * @param p pointer of the param struct
 * @param int_mat target int eigen matrix/vector
 * @param file_name output filename
 * @param sys_log logger of the simulator
 * @return NULL
 * 
 * 
 */
void save_int2bin(const PileParam *p, const int *int_mat, const char* file_name, const logger &sys_log) {
    unsigned int element_num = p->width * p->height;
    FILE* file = fopen(file_name, "wb");
    if (!file) {
        SPDLOG_LOGGER_ERROR(sys_log, "Could not open file: {}", file_name);
        fclose(file);
        return;
    }
    // col-major format
    fwrite(int_mat, sizeof(int), element_num, file);
    fclose(file);
}
