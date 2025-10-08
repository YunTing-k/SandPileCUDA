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
    config_sim_param(sim_p, sim_param_path, sys_log);
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    SPDLOG_LOGGER_INFO(sys_log, "Read simulation parameters done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));
    end_o = std::chrono::steady_clock::now();
    elapsed_o = std::chrono::duration_cast<std::chrono::nanoseconds>(end_o - begin_o);
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[1]: Read configs and configure simulator << done. Time used: {:.3f} ms\n", 1e-6 * (double)(elapsed_o.count()));

    /* Phase[2]-Preparation before simulation */
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[2]: Preparation before simulation << start");
    begin_o = std::chrono::steady_clock::now();
    int *cur_pile, *pile_diff1, *pile_diff2;
    int *pile_host;
    unsigned long long int *count;
    unsigned long long int count_host = 0;
    bool direction = 1;

    cudaMalloc(&cur_pile, pile_p->width * pile_p->height * sizeof(int));
    cudaMalloc(&pile_diff1, pile_p->width * pile_p->height * sizeof(int));
    cudaMalloc(&pile_diff2, pile_p->width * pile_p->height * sizeof(int));
    cudaMalloc(&count, 1 * sizeof(unsigned long long int));
    cudaMemset(cur_pile, 0, pile_p->width * pile_p->height * sizeof(int));
    cudaMemset(pile_diff1, 0, pile_p->width * pile_p->height * sizeof(int));
    cudaMemset(pile_diff2, 0, pile_p->width * pile_p->height * sizeof(int));
    cudaMemset(count, 0, 1 * sizeof(unsigned long long int));
    pile_host = new int[pile_p->width * pile_p->height];

    call_pile_initialize(pile_p, cur_pile);
    cudaMemcpy(pile_host, cur_pile, pile_p->width * pile_p->height * sizeof(int), cudaMemcpyDeviceToHost);
    save_int2bin(pile_p, pile_host, "0.bin", sys_log);
    end_o = std::chrono::steady_clock::now();
    elapsed_o = std::chrono::duration_cast<std::chrono::nanoseconds>(end_o - begin_o);
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[2]: Preparation before simulation << done. Time used: {:.3f} ms\n", 1e-6 * (double)(elapsed_o.count()));

    /* Phase[3]-Simulation */
    switch (pile_p->shape) {
        case TRIANGLE:
            for (int i = 0; i < sim_p->max_itr_steps; i++) {
                begin = std::chrono::steady_clock::now();
                if (direction) {
                    cudaMemset(pile_diff2, 0, pile_p->width * pile_p->height * sizeof(int));
                    call_pile_itr_tri(pile_p, cur_pile, pile_diff1, pile_diff2, count);
                    direction = !direction;
                } else {
                    cudaMemset(pile_diff1, 0, pile_p->width * pile_p->height * sizeof(int));
                    call_pile_itr_tri(pile_p, cur_pile, pile_diff2, pile_diff1, count);
                    direction = !direction;
                }
                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                time += (double)(elapsed.count());
                if ((i + 1) % 10 == 0) {
                    SPDLOG_LOGGER_INFO(sys_log, "Iteration-{} done. Time used: {:.3f} ms", i + 1, 1e-6 * (double)(elapsed.count()));
                    SPDLOG_LOGGER_INFO(sys_log, "++ Average Speed: {:.3f} itr/s. Remaining Time: {:.3f} s\n",
                        (i + 1) / (1e-9 * time),
                        (sim_p->max_itr_steps - i - 1) * (1e-9 * time) / (i + 1));
                }
            }
            cudaMemcpy(pile_host, cur_pile, pile_p->width * pile_p->height * sizeof(int), cudaMemcpyDeviceToHost);
            save_int2bin(pile_p, pile_host, "1_tri.bin", sys_log);
            break;
        case QUADRILATERAL:
            for (int i = 0; i < sim_p->max_itr_steps; i++) {
                begin = std::chrono::steady_clock::now();
                if (direction) {
                    cudaMemset(pile_diff2, 0, pile_p->width * pile_p->height * sizeof(int));
                    call_pile_itr_quad(pile_p, cur_pile, pile_diff1, pile_diff2, count);
                    direction = !direction;
                } else {
                    cudaMemset(pile_diff1, 0, pile_p->width * pile_p->height * sizeof(int));
                    call_pile_itr_quad(pile_p, cur_pile, pile_diff2, pile_diff1, count);
                    direction = !direction;
                }
                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                time += (double)(elapsed.count());
                if ((i + 1) % 10 == 0) {
                    SPDLOG_LOGGER_INFO(sys_log, "Iteration-{} done. Time used: {:.3f} ms", i + 1, 1e-6 * (double)(elapsed.count()));
                    SPDLOG_LOGGER_INFO(sys_log, "++ Average Speed: {:.3f} itr/s. Remaining Time: {:.3f} s\n",
                        (i + 1) / (1e-9 * time),
                        (sim_p->max_itr_steps - i - 1) * (1e-9 * time) / (i + 1));
                }
            }
            cudaMemcpy(pile_host, cur_pile, pile_p->width * pile_p->height * sizeof(int), cudaMemcpyDeviceToHost);
            save_int2bin(pile_p, pile_host, "1_quad.bin", sys_log);
            break;
        case HEXAGON:
            for (int i = 0; i < sim_p->max_itr_steps; i++) {
                begin = std::chrono::steady_clock::now();
                if (direction) {
                    cudaMemset(pile_diff2, 0, pile_p->width * pile_p->height * sizeof(int));
                    call_pile_itr_hex(pile_p, cur_pile, pile_diff1, pile_diff2, count);
                    direction = !direction;
                } else {
                    cudaMemset(pile_diff1, 0, pile_p->width * pile_p->height * sizeof(int));
                    call_pile_itr_hex(pile_p, cur_pile, pile_diff2, pile_diff1, count);
                    direction = !direction;
                }
                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                time += (double)(elapsed.count());
                if ((i + 1) % 10 == 0) {
                    SPDLOG_LOGGER_INFO(sys_log, "Iteration-{} done. Time used: {:.3f} ms", i + 1, 1e-6 * (double)(elapsed.count()));
                    SPDLOG_LOGGER_INFO(sys_log, "++ Average Speed: {:.3f} itr/s. Remaining Time: {:.3f} s\n",
                        (i + 1) / (1e-9 * time),
                        (sim_p->max_itr_steps - i - 1) * (1e-9 * time) / (i + 1));
                }
            }
            cudaMemcpy(pile_host, cur_pile, pile_p->width * pile_p->height * sizeof(int), cudaMemcpyDeviceToHost);
            save_int2bin(pile_p, pile_host, "1_hex.bin", sys_log);
            break;
        default:
            SPDLOG_LOGGER_ERROR(sys_log, "Invalid shape parameter!");
            return -1;
    }
    cudaMemcpy(&count_host, count, 1 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    SPDLOG_LOGGER_INFO(sys_log, "Program done. Sandpile collapse count: {}", count_host);

    /* Phase[5]-Free the all necessary space */
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[5]: Free the all necessary space << start");
    begin_o = std::chrono::steady_clock::now();
    cudaFree(cur_pile);
    cudaFree(pile_diff1);
    cudaFree(pile_diff2);
    cudaFree(count);
    delete[] pile_host;
    end_o = std::chrono::steady_clock::now();
    elapsed_o = std::chrono::duration_cast<std::chrono::nanoseconds>(end_o - begin_o);
    SPDLOG_LOGGER_INFO(sys_log, ">> Phase[5]: Free the all necessary space << done. Time used: {:.3f} ms\n", 1e-6 * (double)(elapsed_o.count()));

    /* final finish up */
    print_end_banner(sys_log);
    spdlog::shutdown();
    return 0;
}
