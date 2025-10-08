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
// ---------------------------------------------------------------------------------
//
//-FHDR//////////////////////////////////////////////////////////////////////////////
#include "utility.h"

using namespace std;
using json = nlohmann::json;

/**
 * Create the logger.
 * 
 * 
 * @param NULL
 * @return NULL
 * 
 * 
 */
logger create_logger() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::debug);

    auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
 
    std::stringstream file_name_ss;
    file_name_ss << std::put_time(std::localtime(&time), "./logs/log_%Y_%m_%d_%H_%M_%S.txt");
    std::string file_name = file_name_ss.str();

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_name.c_str());
    file_sink->set_level(spdlog::level::debug);

    std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
    auto logger = std::make_shared<spdlog::logger>("sys_logger", sinks.begin(), sinks.end());

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
    ifstream file(cfg_path);
    if (!file.is_open()) {
        sys_log->error("Invalid config path!");
        throw runtime_error("Open config file failed");
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
        sys_log->error("Parameters read failed. Exception: {}", e.what());
        throw runtime_error("Parameters read failed");
    }
    file.close();
}
