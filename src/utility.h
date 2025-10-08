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
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <nlohmann/json.hpp>
#include "param.h"

typedef std::shared_ptr<spdlog::logger> logger;

logger create_logger();
void config_param(PARAM *p, const std::string cfg_path, const logger &sys_log);

#endif
