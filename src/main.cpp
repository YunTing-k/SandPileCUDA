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
// ---------------------------------------------------------------------------------
//
//-FHDR//////////////////////////////////////////////////////////////////////////////
#include "utility.h"
#include "pile.h"
#include "data_io.h"

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    std::string config_path, param_path;
    int ret; // return of the function value in int
    int frame_idx = 0; // frame index during simulation
    double time = 0; // time for performance evaluating
    std::chrono::steady_clock::time_point begin, end;
    std::chrono::nanoseconds elapsed;

    /* Get the logger */
    SetConsoleOutputCP(CP_UTF8);
    logger sys_log = create_logger(); // logger

    /* Usage information */
    if (argc != 2) {
        SPDLOG_LOGGER_ERROR(sys_log, "Invalid input! Usage: [exe] <Path of Configs>");
        return -1;
    } else {
        config_path = argv[1];
    }

    /* Read the model parameters from json */
    begin = std::chrono::steady_clock::now();
    PARAM *p;
    p = new PARAM;
    param_path = config_path + "param.json";
    config_param(p, param_path, sys_log);
    end = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    SPDLOG_LOGGER_INFO(sys_log, "Read model parameters done. Time used: {:.3f} ms", 1e-6 * (double)(elapsed.count()));

    /* Simulation */
    int *cur_pile, *pile_diff1, *pile_diff2;
    int *pile_host;
    unsigned long long int *count;
    unsigned long long int count_host = 0;
    bool direction = 1;

    cudaMalloc(&cur_pile, p->width * p->height * sizeof(int));
    cudaMalloc(&pile_diff1, p->width * p->height * sizeof(int));
    cudaMalloc(&pile_diff2, p->width * p->height * sizeof(int));
    cudaMalloc(&count, 1 * sizeof(unsigned long long int));
    cudaMemset(cur_pile, 0, p->width * p->height * sizeof(int));
    cudaMemset(pile_diff1, 0, p->width * p->height * sizeof(int));
    cudaMemset(pile_diff2, 0, p->width * p->height * sizeof(int));
    cudaMemset(count, 0, 1 * sizeof(unsigned long long int));
    pile_host = new int[p->width * p->height];

    call_pile_initialize(p, cur_pile);
    cudaMemcpy(pile_host, cur_pile, p->width * p->height * sizeof(int), cudaMemcpyDeviceToHost);
    save_int2bin(p, pile_host, "0.bin", sys_log);

    switch (p->shape) {
        case TRIANGLE:
            for (int i = 0; i < p->max_itr_steps; i++) {
                begin = std::chrono::steady_clock::now();
                if (direction) {
                    cudaMemset(pile_diff2, 0, p->width * p->height * sizeof(int));
                    call_pile_itr_tri(p, cur_pile, pile_diff1, pile_diff2, count);
                    direction = !direction;
                } else {
                    cudaMemset(pile_diff1, 0, p->width * p->height * sizeof(int));
                    call_pile_itr_tri(p, cur_pile, pile_diff2, pile_diff1, count);
                    direction = !direction;
                }
                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                time += (double)(elapsed.count());
                if ((i + 1) % p->update_steps == 0) {
                    SPDLOG_LOGGER_INFO(sys_log, "Iteration-{} done. Time used: {:.3f} ms", i + 1, 1e-6 * (double)(elapsed.count()));
                    SPDLOG_LOGGER_INFO(sys_log, "++ Average Speed: {:.3f} itr/s. Remaining Time: {:.3f} s\n",
                        (i + 1) / (1e-9 * time),
                        (p->max_itr_steps - i - 1) * (1e-9 * time) / (i + 1));
                }
            }
            cudaMemcpy(pile_host, cur_pile, p->width * p->height * sizeof(int), cudaMemcpyDeviceToHost);
            save_int2bin(p, pile_host, "1_tri.bin", sys_log);
            break;
        case QUADRILATERAL:
            for (int i = 0; i < p->max_itr_steps; i++) {
                begin = std::chrono::steady_clock::now();
                if (direction) {
                    cudaMemset(pile_diff2, 0, p->width * p->height * sizeof(int));
                    call_pile_itr_quad(p, cur_pile, pile_diff1, pile_diff2, count);
                    direction = !direction;
                } else {
                    cudaMemset(pile_diff1, 0, p->width * p->height * sizeof(int));
                    call_pile_itr_quad(p, cur_pile, pile_diff2, pile_diff1, count);
                    direction = !direction;
                }
                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                time += (double)(elapsed.count());
                if ((i + 1) % p->update_steps == 0) {
                    SPDLOG_LOGGER_INFO(sys_log, "Iteration-{} done. Time used: {:.3f} ms", i + 1, 1e-6 * (double)(elapsed.count()));
                    SPDLOG_LOGGER_INFO(sys_log, "++ Average Speed: {:.3f} itr/s. Remaining Time: {:.3f} s\n",
                        (i + 1) / (1e-9 * time),
                        (p->max_itr_steps - i - 1) * (1e-9 * time) / (i + 1));
                }
            }
            cudaMemcpy(pile_host, cur_pile, p->width * p->height * sizeof(int), cudaMemcpyDeviceToHost);
            save_int2bin(p, pile_host, "1_quad.bin", sys_log);
            break;
        case HEXAGON:
            for (int i = 0; i < p->max_itr_steps; i++) {
                begin = std::chrono::steady_clock::now();
                if (direction) {
                    cudaMemset(pile_diff2, 0, p->width * p->height * sizeof(int));
                    call_pile_itr_hex(p, cur_pile, pile_diff1, pile_diff2, count);
                    direction = !direction;
                } else {
                    cudaMemset(pile_diff1, 0, p->width * p->height * sizeof(int));
                    call_pile_itr_hex(p, cur_pile, pile_diff2, pile_diff1, count);
                    direction = !direction;
                }
                end = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
                time += (double)(elapsed.count());
                if ((i + 1) % p->update_steps == 0) {
                    SPDLOG_LOGGER_INFO(sys_log, "Iteration-{} done. Time used: {:.3f} ms", i + 1, 1e-6 * (double)(elapsed.count()));
                    SPDLOG_LOGGER_INFO(sys_log, "++ Average Speed: {:.3f} itr/s. Remaining Time: {:.3f} s\n",
                        (i + 1) / (1e-9 * time),
                        (p->max_itr_steps - i - 1) * (1e-9 * time) / (i + 1));
                }
            }
            cudaMemcpy(pile_host, cur_pile, p->width * p->height * sizeof(int), cudaMemcpyDeviceToHost);
            save_int2bin(p, pile_host, "1_hex.bin", sys_log);
            break;
        default:
            SPDLOG_LOGGER_ERROR(sys_log, "Invalid shape parameter!");
            return -1;
    }
    cudaMemcpy(&count_host, count, 1 * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    SPDLOG_LOGGER_INFO(sys_log, "Program done. Sandpile collapse count: {}", count_host);

    /* free space */
    cudaFree(cur_pile);
    cudaFree(pile_diff1);
    cudaFree(pile_diff2);
    cudaFree(count);
    delete[] pile_host;
    spdlog::shutdown();

    return 0;
}
