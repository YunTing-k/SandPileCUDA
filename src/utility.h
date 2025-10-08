#ifndef UTILITY_H
#define UTILITY_H

#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <windows.h>
#include <cmath>
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <nlohmann/json.hpp>
#include "pile_param.h"
#include "sim_param.h"

#define SandPileMajorVersion 1
#define SandPileMinorVersion 0
#define SandPileUpdateVersion 0

typedef struct SandPileVersion {
    int MajorVersion;
    int MinorVersion;
    int UpdateVersion;
} SandPileVersion;

typedef std::shared_ptr<spdlog::logger> logger;

logger create_logger();
void config_pile_param(PileParam *p, const std::string cfg_path, const logger &sys_log);
void config_sim_param(SimParam *p, const std::string cfg_path, const logger &sys_log);
std::string get_pile_type_name(grid_shape input);
std::string get_frame_sequence_format_name(frame_sequence_format input);
void platform_info(const logger &sys_log);
void print_start_banner(const logger &sys_log);
void print_end_banner(const logger &sys_log);
void get_sandpile_version(SandPileVersion *ver);

#endif
