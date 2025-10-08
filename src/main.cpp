//+FHDR//////////////////////////////////////////////////////////////////////////////
// Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab
// Author: Yu Huang
// Coding: UTF-8
// Create Date: 2025.3.21
// Description: 
// Main program of the Sandpile fractal
//
// Revision:
// ---------------------------------------------------------------------------------
// [Date]         [By]         [Version]         [Change Log]
// ---------------------------------------------------------------------------------
// 2025/03/21     Yu Huang     1.0               First implementation
// 2025/10/08     Yu Huang     1.1               Remove namespace & Adjust logger
// 2025/10/08     Yu Huang     1.2               Add sim params
// 2025/10/08     Yu Huang     1.3               Merge the pile iteration function
// 2025/10/08     Yu Huang     1.4               Video output realization
// 2025/10/08     Yu Huang     1.5               Image output realization
// ---------------------------------------------------------------------------------
//
//-FHDR//////////////////////////////////////////////////////////////////////////////
#include "mimalloc-new-delete.h"
#include "utility.h"
#include "pile.h"
#include "data_io.h"

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    /* Global variable declaration */
    int ret; // function status value in int
    char errbuf[AV_ERROR_MAX_STRING_SIZE] = {0}; // error info buffer
    double time = 0; // time for performance evaluating

    /* Get the logger and display banner */
    #ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
    #endif
    logger sys_log = create_logger();
    print_start_banner(sys_log);
    SandPileVersion pile_ver;
    get_sandpile_version(&pile_ver);
    SPDLOG_LOGGER_INFO(sys_log, "SandPile Fractal Simulator with CUDA");
    SPDLOG_LOGGER_INFO(sys_log, "Simulator Version: {}.{}.{}", pile_ver.MajorVersion, pile_ver.MinorVersion, pile_ver.UpdateVersion);
    SPDLOG_LOGGER_INFO(sys_log, "Copyright (c) 2025-2025 Shanghai Jiao Tong University, Yu Huang\n");

    /* Timer */
    std::chrono::steady_clock::time_point begin, end; // general chrono
    std::chrono::nanoseconds elapsed; // general chrono elapsed time
    std::chrono::steady_clock::time_point begin_o, end_o; // other chrono
    std::chrono::nanoseconds elapsed_o; // other chrono elapsed time
    std::chrono::steady_clock::time_point begin_r, end_r; // row chrono
    std::chrono::nanoseconds elapsed_r; // row chrono elapsed time
    std::chrono::steady_clock::time_point begin_f, end_f; // frame chrono
    std::chrono::nanoseconds elapsed_f; // frame chrono elapsed time
    std::chrono::steady_clock::time_point begin_p, end_p; // postprocess chrono
    std::chrono::nanoseconds elapsed_p; // postprocess chrono elapsed time

    /* Phase[0]-Check platform info */
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[0]: Check platform info << start");
    begin_o = std::chrono::steady_clock::now();

    /* [0]-Platform info */
    platform_info(sys_log);

    /* [0]-Spdlog info */
    SPDLOG_LOGGER_DEBUG(sys_log, "Spdlog version: {}.{}.{}", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR, SPDLOG_VER_PATCH);

    /* [0]-nlohmann/json info */
    SPDLOG_LOGGER_DEBUG(sys_log, "nlohmann/json version: {}.{}.{}",
        NLOHMANN_JSON_VERSION_MAJOR, NLOHMANN_JSON_VERSION_MINOR, NLOHMANN_JSON_VERSION_PATCH);

    /* [0]-Eigen info */
    SPDLOG_LOGGER_DEBUG(sys_log, "Eigen version: {}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);

    /* [0]-FFmpeg info */
    SPDLOG_LOGGER_DEBUG(sys_log, "FFmpeg version: {}", av_version_info());
    SPDLOG_LOGGER_DEBUG(sys_log, " -- avcodec version: {}.{}.{}",
        AV_VERSION_MAJOR(avcodec_version()), AV_VERSION_MINOR(avcodec_version()), AV_VERSION_MICRO(avcodec_version()));
    SPDLOG_LOGGER_DEBUG(sys_log, " -- avformat version: {}.{}.{}",
        AV_VERSION_MAJOR(avformat_version()), AV_VERSION_MINOR(avformat_version()), AV_VERSION_MICRO(avformat_version()));
    SPDLOG_LOGGER_DEBUG(sys_log, " -- swscale version: {}.{}.{}",
        AV_VERSION_MAJOR(swscale_version()), AV_VERSION_MINOR(swscale_version()), AV_VERSION_MICRO(swscale_version()));
    SPDLOG_LOGGER_DEBUG(sys_log, " -- avutil version: {}.{}.{}",
        AV_VERSION_MAJOR(avutil_version()), AV_VERSION_MINOR(avutil_version()), AV_VERSION_MICRO(avutil_version()));
    SPDLOG_LOGGER_DEBUG(sys_log, " -- avfilter version: {}.{}.{}",
        AV_VERSION_MAJOR(avfilter_version()), AV_VERSION_MINOR(avfilter_version()), AV_VERSION_MICRO(avfilter_version()));

    /* [0]-Mimalloc info */
    int mi_ver, mi_ver_major, mi_ver_minor, mi_ver_update;
    mi_ver = mi_version();
    mi_ver_major = mi_ver / 100;
    mi_ver_minor = (mi_ver % 100) / 10;
    mi_ver_update = mi_ver % 10;
    SPDLOG_LOGGER_DEBUG(sys_log, "Mimalloc is used, version: {}.{}.{}", mi_ver_major, mi_ver_minor, mi_ver_update);
    #ifdef DEBUG
    mi_option_set(mi_option_show_stats, 1);
    mi_option_set(mi_option_verbose, 1);
    mi_option_set(mi_option_show_errors, 1);
    #endif

    /* [0]-CUDA info */
    int cuda_ver, cuda_ver_major, cuda_ver_minor, cuda_ver_patch;
    cudaRuntimeGetVersion(&cuda_ver);
    cuda_ver_major = cuda_ver / 1000;
    cuda_ver_minor = (cuda_ver % 100) / 10;
    cuda_ver_patch = cuda_ver % 10;
    SPDLOG_LOGGER_DEBUG(sys_log, "CUDA is used, version: {}.{}.{}", cuda_ver_major, cuda_ver_minor, cuda_ver_patch);
    int device_count = 0;
    int support_managed = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        SPDLOG_LOGGER_WARN(sys_log, "No CUDA devices found");
    } else {
        SPDLOG_LOGGER_DEBUG(sys_log, " -- Available CUDA devices amount: {}", device_count);
        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp device_prop;
            cudaGetDeviceProperties(&device_prop, i);
            cudaDeviceGetAttribute(&support_managed, cudaDevAttrManagedMemory, i);
            SPDLOG_LOGGER_DEBUG(sys_log, " -- Device: {}, {}", i, device_prop.name);
            SPDLOG_LOGGER_DEBUG(sys_log, "    > Compute Capability: {}.{}", device_prop.major, device_prop.minor);
            SPDLOG_LOGGER_DEBUG(sys_log, "    > Total Global Memory: {} MB", device_prop.totalGlobalMem / 1024 / 1024);
            SPDLOG_LOGGER_DEBUG(sys_log, "    > Multiprocessors count: {}", device_prop.multiProcessorCount);
            SPDLOG_LOGGER_DEBUG(sys_log, "    > Max threads per block: {}", device_prop.maxThreadsPerBlock);
            SPDLOG_LOGGER_DEBUG(sys_log, "    > Max threads per multiprocessor: {}", device_prop.maxThreadsPerMultiProcessor);
            SPDLOG_LOGGER_DEBUG(sys_log, "    > Max grid size: [{}, {}, {}]",
                device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
            SPDLOG_LOGGER_DEBUG(sys_log, "    > Max block size: [{}, {}, {}]",
                device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
            if (support_managed == 1) {
                SPDLOG_LOGGER_DEBUG(sys_log, "    > Managed memory support: [YES]");
            } else {
                SPDLOG_LOGGER_DEBUG(sys_log, "    > Managed memory support: [NO]");
            }
        }
        cudaDeviceGetAttribute(&support_managed, cudaDevAttrManagedMemory, 0);
        if (support_managed != 1) {
            SPDLOG_LOGGER_WARN(sys_log, "The default CUDA device doesn't support managed memory");
        }
    }
    cudaSetDevice(0);
    cudaFree(0);
    end_o = std::chrono::steady_clock::now();
    elapsed_o = std::chrono::duration_cast<std::chrono::nanoseconds>(end_o - begin_o);
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[0]: Check platform info << done. Time used: {:.3f} ms\n", 1e-6 * (double)(elapsed_o.count()));

    /* Usage information */
    std::string config_path;
    if (argc != 2) {
        SPDLOG_LOGGER_ERROR(sys_log, "Invalid input! Usage: [exe] <Path of Configs>");
        return -1;
    } else {
        config_path = argv[1];
    }

    /* Phase[1]-Read the config files and configure the simulator */
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[1]: Read configs and configure simulator << start");
    begin_o = std::chrono::steady_clock::now();
    /* [1]-Read the sandpile configs from json */
    begin = std::chrono::steady_clock::now();
    PileParam *pile_p;
    pile_p = new PileParam;
    std::string pile_param_path;
    pile_param_path = config_path + "pile_param.json";
    config_pile_param(pile_p, pile_param_path, sys_log);
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    SPDLOG_LOGGER_INFO(sys_log, "Read sandpile parameters done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));

    /* [1]-Read the simulation configs from json */
    begin = std::chrono::steady_clock::now();
    SimParam *sim_p;
    sim_p = new SimParam;
    std::string sim_param_path;
    sim_param_path = config_path + "sim_param.json";
    config_sim_param(pile_p, sim_p, sim_param_path, sys_log);
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    SPDLOG_LOGGER_INFO(sys_log, "Read simulation parameters done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));
    end_o = std::chrono::steady_clock::now();
    elapsed_o = std::chrono::duration_cast<std::chrono::nanoseconds>(end_o - begin_o);
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[1]: Read configs and configure simulator << done. Time used: {:.3f} ms\n", 1e-6 * (double)(elapsed_o.count()));

    /* Phase[2]-Preparation of data I/O */
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[2]: Preparation for media I/O << start");
    begin_o = std::chrono::steady_clock::now();
    /* [2]-Preparation for output video */
    begin = std::chrono::steady_clock::now();
    AVFormatContext* video_format_ctx = get_outcontext_by_name(sim_p->video_path.c_str(), sys_log);
    const AVCodec* video_codec = get_codec_from_id(AV_CODEC_ID_H264, sys_log);
    /* Buffer allocate */
    AVFrame* buff_frame = av_frame_alloc(); // buffer frame for RGB24 data
    buff_frame->width = pile_p->width;
    buff_frame->height = pile_p->height;
    buff_frame->format = AV_PIX_FMT_RGB24;
    av_frame_get_buffer(buff_frame, 0);
    SPDLOG_LOGGER_DEBUG(sys_log, "Allocate output AVFrame done");
    /* Create new stream */
    AVStream* video_stream = avformat_new_stream(video_format_ctx, video_codec);
    if (!video_stream) {
        SPDLOG_LOGGER_ERROR(sys_log, "Could not create output stream!");
        return -1;
    } // video_stream->id is unique
    SPDLOG_LOGGER_DEBUG(sys_log, "Create output stream succeed, ID: {}", video_stream->index);
    /* Configure the output codec and stream */
    AVCodecContext* video_code_ctx = config_out_codec_ctx(video_codec, sim_p, pile_p, sys_log);
    ret = avcodec_open2(video_code_ctx, video_codec, nullptr);
    if (ret < 0) {
        if (av_strerror(ret, errbuf, sizeof(errbuf)) == 0) {
            SPDLOG_LOGGER_ERROR(sys_log, "Open video codec failed! Error code: {}, error info: {}", ret, errbuf);
        } else {
            SPDLOG_LOGGER_ERROR(sys_log, "Open video codec failed! Error code: {}, error info: Unkown", ret);
        }
        return -1;
    }
    video_stream->codecpar->codec_id = AV_CODEC_ID_H264;
    video_stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    avcodec_parameters_from_context(video_stream->codecpar, video_code_ctx);
    SPDLOG_LOGGER_DEBUG(sys_log, "Configure the output codec and stream succeed");
    /* SWS for RGB24 -> YUV420P */
    SwsContext* yuv_sws_ctx = sws_getContext(
        pile_p->width, pile_p->height, AV_PIX_FMT_RGB24,
        pile_p->width, pile_p->height, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    SPDLOG_LOGGER_DEBUG(sys_log, "Prepare SwsContext [AV_PIX_FMT_RGB24 -> AV_PIX_FMT_YUV420P] succeed");

    AVFrame* video_frame = av_frame_alloc();
    AVPacket* video_packet = av_packet_alloc();
    video_frame->format = video_code_ctx->pix_fmt;
    video_frame->width = video_code_ctx->width;
    video_frame->height = video_code_ctx->height;
    av_frame_get_buffer(video_frame, 0);
    SPDLOG_LOGGER_DEBUG(sys_log, "Allocate output AVFrame done");

    if (!(video_format_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&video_format_ctx->pb, sim_p->video_path.c_str(), AVIO_FLAG_WRITE) < 0) {
            SPDLOG_LOGGER_ERROR(sys_log, "Could not open output file!");
            return -1;
        }
    } else {
        SPDLOG_LOGGER_ERROR(sys_log, "Output format context flag error!");
        return -1;
    }
    ret = avformat_write_header(video_format_ctx, nullptr);
    if (ret < 0) {
        if (av_strerror(ret, errbuf, sizeof(errbuf)) == 0) {
            SPDLOG_LOGGER_ERROR(sys_log, "Write header of the output file failed! Error code: {}, error info: {}", ret, errbuf);
        } else {
            SPDLOG_LOGGER_ERROR(sys_log, "Write header of the output file failed! Error code: {}, error info: Unkown", ret);
        }
        return -1;
    } else {
        SPDLOG_LOGGER_DEBUG(sys_log, "Write header succeed with return value: {}", ret);
    }
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    SPDLOG_LOGGER_INFO(sys_log, "Preparation for output video done. Time used: {:.3f} ms\n", 1e-6 * (double)(elapsed.count()));

    /* [2]-output frame sequence codec and context */
    begin = std::chrono::steady_clock::now();
    int outframe_seq_idx = 0; // output frame sequence index during visualization
    std::string outseq_file_name;
    std::string outseq_fmt;
    AVCodecContext* outseq_code_ctx;
    AVCodecID outseq_codec_id;
    AVPixelFormat outseq_pix_format;
    switch (sim_p->outseq_format) {
        case SEQUENCE_JPEG:
            outseq_codec_id = AV_CODEC_ID_MJPEG;
            outseq_pix_format = AV_PIX_FMT_YUVJ420P;
            outseq_fmt = ".jpg";
            SPDLOG_LOGGER_WARN(sys_log, "Output frame sequence: JPEG format may suffer from obvious loss in encoding");
            break;
        case SEQUENCE_PNG:
            outseq_codec_id = AV_CODEC_ID_PNG;
            outseq_pix_format = AV_PIX_FMT_RGB24;
            outseq_fmt = ".png";
            break;
        case SEQUENCE_TIFF:
            outseq_codec_id = AV_CODEC_ID_TIFF;
            outseq_pix_format = AV_PIX_FMT_RGB24;
            outseq_fmt = ".tiff";
            break;
        case SEQUENCE_BMP:
            outseq_codec_id = AV_CODEC_ID_BMP;
            outseq_pix_format = AV_PIX_FMT_BGR24;
            outseq_fmt = ".bmp";
            break;
        case SEQUENCE_JPEG2000:
            outseq_codec_id = AV_CODEC_ID_JPEG2000;
            outseq_pix_format = AV_PIX_FMT_RGB24;
            outseq_fmt = ".jp2";
            break;
        default:
            SPDLOG_LOGGER_ERROR(sys_log, "Invalid frame sequence format!");
            return -1;
    }
    const AVCodec* outseq_codec = get_codec_from_id(outseq_codec_id, sys_log);
    outseq_code_ctx = avcodec_alloc_context3(outseq_codec);
    outseq_code_ctx->width = pile_p->width;
    outseq_code_ctx->height = pile_p->height;
    outseq_code_ctx->time_base.num = 1;
    outseq_code_ctx->time_base.den = 1;
    outseq_code_ctx->thread_count = sim_p->thread_count;
    outseq_code_ctx->pix_fmt = outseq_pix_format;
    if (avcodec_open2(outseq_code_ctx, outseq_codec, nullptr) < 0) {
        SPDLOG_LOGGER_ERROR(sys_log, "Could not open output frame sequence codec!");
        return -1;
    }
    SPDLOG_LOGGER_DEBUG(sys_log, "Get output frame sequence codec and allocate context done");

    SwsContext* outseq_sws_ctx;
    AVFrame* outseq_frame;
    if (outseq_pix_format != AV_PIX_FMT_RGB24) {
        outseq_sws_ctx = sws_getContext(
        pile_p->width, pile_p->height, AV_PIX_FMT_RGB24,
        pile_p->width, pile_p->height, outseq_pix_format,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
        SPDLOG_LOGGER_DEBUG(sys_log, "Prepare SwsContext [AV_PIX_FMT_RGB24 -> {}] succeed", av_get_pix_fmt_name(outseq_pix_format));

        outseq_frame = av_frame_alloc();
        outseq_frame->width = pile_p->width;
        outseq_frame->height = pile_p->height;
        outseq_frame->format = outseq_pix_format;
        av_frame_get_buffer(outseq_frame, 0);
        SPDLOG_LOGGER_DEBUG(sys_log, "Allocate output frame sequence frame done");
    }

    AVPacket* outseq_packet = av_packet_alloc();
    int outseq_buff_size = av_image_get_buffer_size(outseq_pix_format, pile_p->width, pile_p->height, 1);
    uint8_t* outseq_buffer = (uint8_t*)av_malloc(outseq_buff_size);
    SPDLOG_LOGGER_DEBUG(sys_log, "Allocate output frame sequence buffer done");
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    SPDLOG_LOGGER_INFO(sys_log, "Preparation for output frame sequence done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));
    end_o = std::chrono::steady_clock::now();
    elapsed_o = std::chrono::duration_cast<std::chrono::nanoseconds>(end_o - begin_o);
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[2]: Preparation for media I/O << done. Time used: {:.3f} ms\n", 1e-6 * (double)(elapsed_o.count()));

    /* Phase[3]-Preparation before simulation */
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[3]: Preparation before simulation << start");
    begin_o = std::chrono::steady_clock::now();

    Eigen::initParallel();
    Eigen::setNbThreads(sim_p->max_threads);
    omp_set_num_threads(sim_p->max_threads);
    SPDLOG_LOGGER_DEBUG(sys_log, "Eigen configured with using {} threads", Eigen::nbThreads());
    #pragma omp parallel
    {
        #pragma omp single
        {
            SPDLOG_LOGGER_DEBUG(sys_log, "OpenMP configured with using {} threads", omp_get_num_threads());
        }
    }
    SPDLOG_LOGGER_DEBUG(sys_log, "OpenMP nested: {}", omp_get_nested());

    SandPileContextHost ctx_host;
    SandPileContextDevice ctx_device;
    bool direction = 1;

    ctx_host.pile_host = new int[pile_p->width * pile_p->height];
    ctx_host.count = 0;
    for (int i = 0; i < 6; i++) {
        ctx_host.lut_r[i] = sim_p->lut_r[i];
        ctx_host.lut_g[i] = sim_p->lut_g[i];
        ctx_host.lut_b[i] = sim_p->lut_b[i];
    }
    prepare_rgb_mat(pile_p->height, pile_p->width, ctx_host.r_mat, ctx_host.g_mat, ctx_host.b_mat);
    cudaMalloc(&ctx_device.pile_device, pile_p->width * pile_p->height * sizeof(int));
    cudaMalloc(&ctx_device.pile_diff1, pile_p->width * pile_p->height * sizeof(int));
    cudaMalloc(&ctx_device.pile_diff2, pile_p->width * pile_p->height * sizeof(int));
    cudaMalloc(&ctx_device.count, 1 * sizeof(unsigned long long int));
    cudaMemset(ctx_device.pile_device, 0, pile_p->width * pile_p->height * sizeof(int));
    cudaMemset(ctx_device.pile_diff1, 0, pile_p->width * pile_p->height * sizeof(int));
    cudaMemset(ctx_device.pile_diff2, 0, pile_p->width * pile_p->height * sizeof(int));
    cudaMemset(ctx_device.count, 0, 1 * sizeof(unsigned long long int));
    cudaMalloc(&ctx_device.lut_r, 6 * sizeof(int));
    cudaMalloc(&ctx_device.lut_g, 6 * sizeof(int));
    cudaMalloc(&ctx_device.lut_b, 6 * sizeof(int));
    cudaMemcpy(ctx_device.lut_r, ctx_host.lut_r, 6 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx_device.lut_g, ctx_host.lut_g, 6 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx_device.lut_b, ctx_host.lut_b, 6 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&ctx_device.r_mat, pile_p->width * pile_p->height * sizeof(int));
    cudaMalloc(&ctx_device.g_mat, pile_p->width * pile_p->height * sizeof(int));
    cudaMalloc(&ctx_device.b_mat, pile_p->width * pile_p->height * sizeof(int));

    call_pile_initialize(pile_p, ctx_device.pile_device);
    cudaMemcpy(ctx_host.pile_host, ctx_device.pile_device, pile_p->width * pile_p->height * sizeof(int), cudaMemcpyDeviceToHost);
    end_o = std::chrono::steady_clock::now();
    elapsed_o = std::chrono::duration_cast<std::chrono::nanoseconds>(end_o - begin_o);
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[3]: Preparation before simulation << done. Time used: {:.3f} ms\n", 1e-6 * (double)(elapsed_o.count()));

    /* Phase[4]-Main process of simulation */
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[4]: Main process of simulation << start");
    begin_o = std::chrono::steady_clock::now();
    begin_f = std::chrono::steady_clock::now();
    int frame_idx = 0;
    for (int i = 0; i < sim_p->max_itr_steps; i++) {
        if (direction) {
            cudaMemset(ctx_device.pile_diff2, 0, pile_p->width * pile_p->height * sizeof(int));
            pile_itr(pile_p, ctx_device.pile_device, ctx_device.pile_diff1, ctx_device.pile_diff2, ctx_device.count, sys_log);
            direction = !direction;
        } else {
            cudaMemset(ctx_device.pile_diff1, 0, pile_p->width * pile_p->height * sizeof(int));
            pile_itr(pile_p, ctx_device.pile_device, ctx_device.pile_diff2, ctx_device.pile_diff1, ctx_device.count, sys_log);
            direction = !direction;
        }
        if ((i + 1) % sim_p->sp_rate == 0) {
            /* simulation info */
            end_f = std::chrono::steady_clock::now();
            elapsed_f = std::chrono::duration_cast<std::chrono::nanoseconds>(end_f - begin_f);
            time += (double)(elapsed_f.count());
            SPDLOG_LOGGER_INFO(sys_log, " -- Iteration-{} done. Time used: {:.3f} ms", i + 1, 1e-6 * (double)(elapsed_f.count()));
            SPDLOG_LOGGER_INFO(sys_log, "    > Average Speed: {:.3f} itr/s. Remaining Time(simulation only): {:.3f} s",
                (i + 1) / (1e-9 * time), (sim_p->max_itr_steps - i - 1) * (1e-9 * time) / (i + 1));

            /* visualization */
            begin = std::chrono::steady_clock::now();
            if (sim_p->visualize_cuda) {
                call_visualize_cuda(pile_p, ctx_device);
            } else {
                cudaMemcpy(ctx_host.pile_host, ctx_device.pile_device, pile_p->width * pile_p->height * sizeof(int), cudaMemcpyDeviceToHost);
                visualize_host(pile_p, ctx_host);
            }
            end = std::chrono::steady_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            SPDLOG_LOGGER_INFO(sys_log, "    > Visualization done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));

            /* frame transformation */
            begin = std::chrono::steady_clock::now();
            if (sim_p->visualize_cuda) { // fetch data from device
                cudaMemcpy(ctx_host.r_mat.data(), ctx_device.r_mat, pile_p->width * pile_p->height * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(ctx_host.g_mat.data(), ctx_device.g_mat, pile_p->width * pile_p->height * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(ctx_host.b_mat.data(), ctx_device.b_mat, pile_p->width * pile_p->height * sizeof(int), cudaMemcpyDeviceToHost);
            }
            get_rgb_frame(ctx_host.r_mat, ctx_host.g_mat, ctx_host.b_mat, buff_frame, sys_log); // R/G/B mat to AVFrame
            end = std::chrono::steady_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            SPDLOG_LOGGER_INFO(sys_log, "    > Frame transformation done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));

            /* video encoding */
            sws_scale(yuv_sws_ctx, buff_frame->data, buff_frame->linesize, 0, pile_p->height, video_frame->data, video_frame->linesize);
            video_frame->pts = frame_idx;
            ret = avcodec_send_frame(video_code_ctx, video_frame);
            if (ret == 0) {
                // make sure all the packets are obtained, in some codec, need to send multiple frames before get one packet
                while(avcodec_receive_packet(video_code_ctx, video_packet) == 0) {
                    video_packet->stream_index = video_stream->index;
                    av_packet_rescale_ts(video_packet, video_code_ctx->time_base, video_stream->time_base);
                    av_interleaved_write_frame(video_format_ctx, video_packet);
                    av_packet_unref(video_packet);
                }
                av_packet_unref(video_packet);
            } else {
                if (av_strerror(ret, errbuf, sizeof(errbuf)) == 0) {
                    SPDLOG_LOGGER_ERROR(sys_log, "Could not send frame to video codec! Error code: {}, error info: {}", ret, errbuf);
                } else {
                    SPDLOG_LOGGER_ERROR(sys_log, "Could not send frame to video codec! Error code: {}, error info: Unkown", ret);
                }
                return -1;
            }
            end = std::chrono::steady_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
            SPDLOG_LOGGER_INFO(sys_log, " -- Frame encoding done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));

            /* raw data export */
            if (sim_p->save_bin) {
                begin = std::chrono::steady_clock::now();
                std::string bin_file_path = sim_p->data_path + "bin/" + std::to_string(frame_idx) + ".bin";
                save_int2bin(pile_p, ctx_host.pile_host, bin_file_path.c_str(), sys_log);
                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                SPDLOG_LOGGER_INFO(sys_log, " -- Raw data in bin saved. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));
            }

            /* frame sequence */
            if (sim_p->save_seq) {
                begin = std::chrono::steady_clock::now();
                if (outseq_pix_format != AV_PIX_FMT_RGB24) {
                    sws_scale(outseq_sws_ctx, buff_frame->data, buff_frame->linesize, 0, pile_p->height, outseq_frame->data, outseq_frame->linesize);
                    ret = avcodec_send_frame(outseq_code_ctx, outseq_frame);
                } else {
                    ret = avcodec_send_frame(outseq_code_ctx, buff_frame);
                }
                if (ret == 0) {
                    outseq_file_name = sim_p->data_path + "img/frame_" + std::to_string(outframe_seq_idx + 1) + outseq_fmt;
                    FILE* outseq_file = fopen(outseq_file_name.c_str(), "wb");
                    if (!outseq_file) {
                        SPDLOG_LOGGER_ERROR(sys_log, "Could not open file for frame sequence: {}", outseq_file_name);
                        return -1;
                    }
                    ret = avcodec_receive_packet(outseq_code_ctx, outseq_packet);
                    if (ret == 0) {
                        memcpy(outseq_buffer, (*outseq_packet).data, (*outseq_packet).size);
                        fwrite(outseq_buffer, sizeof(uint8_t), (*outseq_packet).size, outseq_file);
                        outframe_seq_idx ++;
                    }
                    fclose(outseq_file);
                } else {
                    if (av_strerror(ret, errbuf, sizeof(errbuf)) == 0) {
                        SPDLOG_LOGGER_ERROR(sys_log, "Failed to send frame to output frame sequence codec! Error code! Error code: {}, error info: {}", ret, errbuf);
                    } else {
                        SPDLOG_LOGGER_ERROR(sys_log, "Failed to send frame to output frame sequence codec! Error code! Error code: {}, error info: Unkown", ret);
                    }
                    return -1;
                }
                av_packet_unref(outseq_packet);
                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                SPDLOG_LOGGER_INFO(sys_log, " -- Output frame sequence transform and encoding done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));
            }

            /* start next timer */
            frame_idx++;
            begin_f = std::chrono::steady_clock::now();
        }
    }
    cudaMemcpy(ctx_host.pile_host, ctx_device.pile_device, pile_p->width * pile_p->height * sizeof(int), cudaMemcpyDeviceToHost);
    std::string data_file_path = sim_p->data_path + "bin/final.bin";
    save_int2bin(pile_p, ctx_host.pile_host, data_file_path.c_str(), sys_log);
    SPDLOG_LOGGER_INFO(sys_log, "Final sandpile data in bin saved in: {}", data_file_path);
    cudaMemcpy(&ctx_host.count, ctx_device.count, 1 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    SPDLOG_LOGGER_INFO(sys_log, "Simulation done. Sandpile collapse count: {}", ctx_host.count);
    end_o = std::chrono::steady_clock::now();
    elapsed_o = std::chrono::duration_cast<std::chrono::nanoseconds>(end_o - begin_o);
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[4]: Main process of simulation << done. Time used: {:.3f} ms, {:.3f} frame/ms\n",
        1e-6 * (double)(elapsed_o.count()), sim_p->max_itr_steps / (1e-6 * (double)(elapsed_o.count())));

    /* Phase[5] Finish up of media I/O */
    begin_o = std::chrono::steady_clock::now();
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[5]: Finish up of media I/O << start");
    /* [5]-finish up of video encoding */
    begin = std::chrono::steady_clock::now();
    SPDLOG_LOGGER_INFO(sys_log, "Start the finish up of video encoding");
    // flush codec
    ret = avcodec_send_frame(video_code_ctx, nullptr);
    if (ret < 0) {
        if (av_strerror(ret, errbuf, sizeof(errbuf)) == 0) {
            SPDLOG_LOGGER_ERROR(sys_log, "Could not flush the video codec! Error code: {}, error info: {}", ret, errbuf);
        } else {
            SPDLOG_LOGGER_ERROR(sys_log, "Could not flush the video codec! Error code: {}, error info: Unkown", ret);
        }
        return -1;
    }
    while (ret >= 0) {
        ret = avcodec_receive_packet(video_code_ctx, video_packet);
        if (ret == 0) {
            video_packet->stream_index = video_stream->index;
            av_packet_rescale_ts(video_packet, video_code_ctx->time_base, video_stream->time_base);
            av_interleaved_write_frame(video_format_ctx, video_packet);
            av_packet_unref(video_packet);
        }
        av_packet_unref(video_packet);
    }
    // write trailer
    ret = av_write_trailer(video_format_ctx);
    if (ret < 0) {
        if (av_strerror(ret, errbuf, sizeof(errbuf)) == 0) {
            SPDLOG_LOGGER_ERROR(sys_log, "Can not write trailer! Error code: {}, error info: {}", ret, errbuf);
        } else {
            SPDLOG_LOGGER_ERROR(sys_log, "Can not write trailer! Error code: {}, error info: Unkown", ret);
        }
        return -1;
    }
    SPDLOG_LOGGER_INFO(sys_log, "Write trailer succeed with return value: {}", ret);
    if (!(video_format_ctx->oformat->flags & AVFMT_NOFILE)) {
        if (avio_closep(&video_format_ctx->pb) < 0) {
            SPDLOG_LOGGER_ERROR(sys_log, "Could not close output media file!");
        } else {
            SPDLOG_LOGGER_INFO(sys_log, "Output media file closed");
        }
    } else {
        SPDLOG_LOGGER_ERROR(sys_log, "Output format context flag error!");
    }
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    SPDLOG_LOGGER_INFO(sys_log, "Finish up of video encoding done. Time used: {:.3f} ms\n", 1e-6 * (double)(elapsed.count()));

    /* [5]-finish up of output frame transform and sequence encoding */
    if (sim_p->save_seq) {
        begin = std::chrono::steady_clock::now();
        SPDLOG_LOGGER_INFO(sys_log, "Start the finish up of output frame sequence encoding");
        ret = avcodec_send_frame(outseq_code_ctx, nullptr);
        if (ret < 0) {
            if (av_strerror(ret, errbuf, sizeof(errbuf)) == 0) {
                SPDLOG_LOGGER_ERROR(sys_log, "Could not flush the output frame sequence codec! Error code: {}, error info: {}", ret, errbuf);
            } else {
                SPDLOG_LOGGER_ERROR(sys_log, "Could not flush the output frame sequence codec! Error code: {}, error info: Unkown", ret);
            }
            return -1;
        }
        while (ret >= 0) {
            ret = avcodec_receive_packet(outseq_code_ctx, outseq_packet);
            if (ret == 0) {
                outseq_file_name = sim_p->data_path + "img/frame_" + std::to_string(outframe_seq_idx + 1) + outseq_fmt;
                FILE* outseq_file = fopen(outseq_file_name.c_str(), "wb");
                if (!outseq_file) {
                    SPDLOG_LOGGER_ERROR(sys_log, "Could not open file for frame sequence: {}", outseq_file_name);
                    return -1;
                }
                memcpy(outseq_buffer, (*outseq_packet).data, (*outseq_packet).size);
                fwrite(outseq_buffer, sizeof(uint8_t), (*outseq_packet).size, outseq_file);
                outframe_seq_idx ++;
                fclose(outseq_file);
            }
            av_packet_unref(outseq_packet);
        }
        end = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        SPDLOG_LOGGER_INFO(sys_log, "Finish up of output frame sequence encoding done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));
    }
    end_o = std::chrono::steady_clock::now();
    elapsed_o = std::chrono::duration_cast<std::chrono::nanoseconds>(end_o - begin_o);
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[5]: Finish up of media I/O << done. Time used: {:.3f} ms\n", 1e-6 * (double)(elapsed_o.count()));

    /* Phase[6]-Free the all necessary space */
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[6]: Free the all necessary space << start");
    begin_o = std::chrono::steady_clock::now();
    /* [6]-free space of FFmpeg */
    begin = std::chrono::steady_clock::now();
    SPDLOG_LOGGER_INFO(sys_log, "Start freeing space of FFmpeg");
    av_frame_free(&buff_frame);
    avformat_free_context(video_format_ctx);
    avcodec_free_context(&video_code_ctx);
    sws_freeContext(yuv_sws_ctx);
    av_frame_free(&video_frame);
    av_packet_free(&video_packet);
    avcodec_free_context(&outseq_code_ctx);
    av_packet_free(&outseq_packet);
    av_free(outseq_buffer);
    if (outseq_pix_format != AV_PIX_FMT_RGB24) {
        sws_freeContext(outseq_sws_ctx);
        av_frame_free(&outseq_frame);
    }
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    SPDLOG_LOGGER_INFO(sys_log, "Freeing space of FFmpeg done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));

    /* [6]-free space of CUDA */
    begin = std::chrono::steady_clock::now();
    SPDLOG_LOGGER_INFO(sys_log, "Start freeing space of CUDA objects");
    cudaFree(ctx_device.pile_device);
    cudaFree(ctx_device.pile_diff1);
    cudaFree(ctx_device.pile_diff2);
    cudaFree(ctx_device.count);
    cudaFree(ctx_device.lut_r);
    cudaFree(ctx_device.lut_g);
    cudaFree(ctx_device.lut_b);
    cudaFree(ctx_device.r_mat);
    cudaFree(ctx_device.g_mat);
    cudaFree(ctx_device.b_mat);
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    SPDLOG_LOGGER_INFO(sys_log, "Freeing space of CUDA objects done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));

    /* [6]-free space of other params */
    begin = std::chrono::steady_clock::now();
    SPDLOG_LOGGER_INFO(sys_log, "Start freeing space of other objects");
    delete[] ctx_host.pile_host;
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    SPDLOG_LOGGER_INFO(sys_log, "Freeing space of other objects done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));

    end_o = std::chrono::steady_clock::now();
    elapsed_o = std::chrono::duration_cast<std::chrono::nanoseconds>(end_o - begin_o);
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[6]: Free the all necessary space << done. Time used: {:.3f} ms\n", 1e-6 * (double)(elapsed_o.count()));

    /* final finish up */
    print_end_banner(sys_log);
    spdlog::shutdown();
    return 0;
}
