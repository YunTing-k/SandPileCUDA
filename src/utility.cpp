//+FHDR//////////////////////////////////////////////////////////////////////////////
// Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab
// Author: Yu Huang
// Coding: UTF-8
// Create Date: 2025.2.28
// Description: 
// Tools and utility for simulator and console.
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

using json = nlohmann::json;

/**
 * Create the logger.
 * 
 * 
 * @param NULL
 * @return logger, the spd logger of the TECoSim simulator
 * 
 * 
 */
logger create_logger() {
    // console sink configure
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::debug);
    console_sink->set_pattern("\033[90m[%Y-%m-%d %H:%M:%S.%e]\033[0m [%^%l%$] %v");
    // file sink configure
    auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream file_name_ss;
    file_name_ss << std::put_time(std::localtime(&time), "./logs/log_%Y_%m_%d_%H_%M_%S.txt");
    std::string file_name = file_name_ss.str();
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_name.c_str());
    file_sink->set_level(spdlog::level::trace);
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [proc:%P thd:%t] [%s:%#->%!] [%l] %v");
    // define logger
    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    auto logger = std::make_shared<spdlog::logger>("sys_logger", sinks.begin(), sinks.end());
    // configure logger
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::debug);
    spdlog::flush_on(spdlog::level::warn);
    spdlog::flush_every(std::chrono::seconds(1));
    return logger;
}

/**
 * Config the model param from json file.
 * 
 * 
 * @param p pointer of the param struct
 * @param cfg_path path of the param config path
 * @param sys_log logger of the simulator
 * @return NULL
 * 
 * 
 */
void config_param(PARAM *p, const std::string cfg_path, const logger &sys_log) {
    json cfg;
    std::ifstream file(cfg_path);
    if (!file.is_open()) {
        SPDLOG_LOGGER_ERROR(sys_log, "Invalid config path!");
        throw std::runtime_error("Open config file failed");
    }
    cfg = json::parse(file);

    try {
        p->shape = (grid_shape)cfg.at("shape");
        p->width = cfg.at("width");
        p->height = cfg.at("height");
        p->ini_sand_num = cfg.at("ini_sand_num");
        p->max_itr_steps = cfg.at("max_itr_steps");
        p->update_steps = cfg.at("update_steps");
        p->sp_rate = cfg.at("sp_rate");
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(sys_log, "Parameters read failed. Exception: {}", e.what());
        throw std::runtime_error("Parameters read failed");
    }
    file.close();
}
