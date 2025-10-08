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
// 2025/10/08     Yu Huang     1.3               Video output realization
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
    SPDLOG_LOGGER_INFO(sys_log, "Open video source:{} succeed", file_name);
    SPDLOG_LOGGER_INFO(sys_log, " -- URL: {}", video_format_ctx->url);
    SPDLOG_LOGGER_INFO(sys_log, " -- Number of streams: {}", video_format_ctx->nb_streams);
    SPDLOG_LOGGER_INFO(sys_log, " -- Start time: {:.3f} s", (double)(video_format_ctx->start_time / AV_TIME_BASE));
    SPDLOG_LOGGER_INFO(sys_log, " -- Duration time: {:.3f} s", (double)(video_format_ctx->duration / AV_TIME_BASE));
    SPDLOG_LOGGER_INFO(sys_log, " -- Bit rate: {} bit/s", video_format_ctx->bit_rate);
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
    AVRational avg_frame_rate, r_frame_rate;
    AVPixelFormat format;
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
    avg_frame_rate = video_format_ctx->streams[video_stream_idx]->avg_frame_rate;
    r_frame_rate = video_format_ctx->streams[video_stream_idx]->r_frame_rate;
    format = (AVPixelFormat)video_format_ctx->streams[video_stream_idx]->codecpar->format;
    SPDLOG_LOGGER_INFO(sys_log, "Find video stream succeed");
    SPDLOG_LOGGER_INFO(sys_log, " -- Video stream index: {}", video_stream_idx);
    SPDLOG_LOGGER_INFO(sys_log, " -- Video stream average frame rate: {:.3f}", (double)avg_frame_rate.num / (double)avg_frame_rate.den);
    SPDLOG_LOGGER_INFO(sys_log, " -- Video stream real frame rate: {:.3f}", (double)r_frame_rate.num / (double)r_frame_rate.den);
    SPDLOG_LOGGER_INFO(sys_log, " -- Video resolution: [{}, {}]",
         video_format_ctx->streams[video_stream_idx]->codecpar->width,
         video_format_ctx->streams[video_stream_idx]->codecpar->height);
    SPDLOG_LOGGER_INFO(sys_log, " -- Video pixel format: {}", av_get_pix_fmt_name(format));
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
    SPDLOG_LOGGER_DEBUG(sys_log, "Get codec from video stream succeed");
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
    SPDLOG_LOGGER_DEBUG(sys_log, "Get codec from codec ID: {} succeed", avcodec_get_name(in_codec_id));
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
    SPDLOG_LOGGER_DEBUG(sys_log, "Allocate codec context and open the codec succeed");
    return codec_ctx;
}

/**
 * get the output context by filename.
 * 
 * 
 * @param file_name file name that you wish to save the output contents, e.g., xx.mp4
 * @param sys_log logger of the simulator
 * @return output AVFormatContext by filename.
 * 
 * 
 */
AVFormatContext* get_outcontext_by_name(const char* file_name, const logger &sys_log) {
    AVFormatContext* out_format_ctx = nullptr;
    avformat_alloc_output_context2(&out_format_ctx, nullptr, nullptr, file_name);
    if (!out_format_ctx) {
        SPDLOG_LOGGER_ERROR(sys_log, "Output context allocate failed!");
        throw std::runtime_error("Could not create output context");
    }
    SPDLOG_LOGGER_DEBUG(sys_log, "Get the output context by filename {} succeed", file_name);
    return out_format_ctx;
}

/**
 * config the output codec context.
 * 
 * 
 * @param in_codec input target codec
 * @param sim_p pointer of the simulation param struct
 * @param panel_p pointer of the panel param struct
 * @param sys_log logger of the simulator
 * @return output AVCodecContext after configuration.
 * 
 * 
 */
AVCodecContext* config_out_codec_ctx(const AVCodec* in_codec, const SimParam* sim_p, const PileParam* pile_p, const logger &sys_log) {
    AVCodecContext* out_codec_ctx = avcodec_alloc_context3(in_codec);
    out_codec_ctx->bit_rate = sim_p->bit_rate;
    // out_codec_ctx->bit_rate = 8000000;
    out_codec_ctx->rc_max_rate = sim_p->rc_max_rate;
    out_codec_ctx->rc_min_rate = sim_p->rc_min_rate;
    out_codec_ctx->rc_buffer_size = sim_p->rc_buffer_size;
    out_codec_ctx->width = pile_p->width;
    out_codec_ctx->height = pile_p->height;
    out_codec_ctx->time_base = {1, sim_p->fresh_rate};
    out_codec_ctx->gop_size = sim_p->gop_size;
    // out_codec_ctx->gop_size = 10;
    out_codec_ctx->max_b_frames = sim_p->max_b_frames;
    // out_codec_ctx->max_b_frames = 0;
    out_codec_ctx->thread_count = sim_p->thread_count;
    // out_codec_ctx->thread_count = 0;
    out_codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    av_opt_set(out_codec_ctx->priv_data, "preset", "veryslow", 0);
    av_opt_set(out_codec_ctx->priv_data, "tune", "film", 0);
    av_opt_set(out_codec_ctx->priv_data, "crf", "18", 0);

    SPDLOG_LOGGER_INFO(sys_log, "Output codec context configured done");
    SPDLOG_LOGGER_INFO(sys_log, " -- Codec ID: {}, Name: {}", (int)(in_codec->id), in_codec->name);
    SPDLOG_LOGGER_INFO(sys_log, " -- Codec longname: {}", in_codec->long_name);
    SPDLOG_LOGGER_INFO(sys_log, " -- Bit rate: {} bit/s", out_codec_ctx->bit_rate);
    SPDLOG_LOGGER_INFO(sys_log, " -- Maximum bit rate: {} bit/s", out_codec_ctx->rc_max_rate);
    SPDLOG_LOGGER_INFO(sys_log, " -- Minimum bit rate: {} bit/s", out_codec_ctx->rc_min_rate);
    SPDLOG_LOGGER_INFO(sys_log, " -- Codec bitstream buffer size: {} bits", out_codec_ctx->rc_buffer_size);
    SPDLOG_LOGGER_INFO(sys_log, " -- Output resolution: [{}, {}]", out_codec_ctx->width, out_codec_ctx->height);
    SPDLOG_LOGGER_INFO(sys_log, " -- Output FPS: {:.3f}", (double)out_codec_ctx->time_base.den / (double)out_codec_ctx->time_base.num);
    SPDLOG_LOGGER_INFO(sys_log, " -- Codec gop size: {}", out_codec_ctx->gop_size);
    SPDLOG_LOGGER_INFO(sys_log, " -- Codec maximum b frames: {}", out_codec_ctx->max_b_frames);
    SPDLOG_LOGGER_INFO(sys_log, " -- Codec thread count: {}", out_codec_ctx->thread_count);
    SPDLOG_LOGGER_INFO(sys_log, " -- Output pixel format: {}", av_get_pix_fmt_name(AV_PIX_FMT_YUV420P));

    return out_codec_ctx;
}

/**
 * Build the R/G/B matrix in eigen.
 * 
 * 
 * @param height panel pixel height, rows, M
 * @param width panel pixel width, cols, N
 * @param r_mat, g_mat, b_mat reference of the R/G/B matrix
 * @return NULL
 * 
 * 
 */
void prepare_rgb_mat(int height, int width, Eigen::MatrixXi &r_mat, Eigen::MatrixXi &g_mat, Eigen::MatrixXi &b_mat) {
    r_mat.resize(height, width);r_mat.setZero();
    g_mat.resize(height, width);g_mat.setZero();
    b_mat.resize(height, width);b_mat.setZero();
}

/**
 * get the AVFrame in RGB24 format from R/G/B mat in dense matrix from Eigen.
 * 
 * 
 * @param r_mat, g_mat, b_mat reference of the R/G/B matrix
 * @param frame output RGB24 format frame
 * @param sys_log logger of the simulator
 * @return NULL
 * 
 * 
 */
void get_rgb_frame(const Eigen::MatrixXi &r_mat, const Eigen::MatrixXi &g_mat, const Eigen::MatrixXi &b_mat, AVFrame* frame, const logger &sys_log) {
    if (frame->height != r_mat.rows() || frame->width != r_mat.cols()) {
        SPDLOG_LOGGER_ERROR(sys_log, "Input Red mat size [{}, {}] unmatched with output frame [{}, {}]", r_mat.cols(), r_mat.rows(), frame->width, frame->height);
        throw std::runtime_error("Frame size unmatched with Red mat");
    }
    if (frame->height != g_mat.rows() || frame->width != g_mat.cols()) {
        SPDLOG_LOGGER_ERROR(sys_log, "Input Green mat size [{}, {}] unmatched with output frame [{}, {}]", g_mat.cols(), g_mat.rows(), frame->width, frame->height);
        throw std::runtime_error("Frame size unmatched with Green mat");
    }
    if (frame->height != b_mat.rows() || frame->width != b_mat.cols()) {
        SPDLOG_LOGGER_ERROR(sys_log, "Input Blue mat size [{}, {}] unmatched with output frame [{}, {}]", b_mat.cols(), b_mat.rows(), frame->width, frame->height);
        throw std::runtime_error("Frame size unmatched with Blue mat");
    }
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < frame->height; row++) {
        for (int col = 0; col < frame->width; col++) {
            *(frame->data[0] + row * frame->linesize[0] + 3 * col + 0) = r_mat(row, col);
            *(frame->data[0] + row * frame->linesize[0] + 3 * col + 1) = g_mat(row, col);
            *(frame->data[0] + row * frame->linesize[0] + 3 * col + 2) = b_mat(row, col);
        }
    }
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
