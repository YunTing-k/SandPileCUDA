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
// 2025/10/08     Yu Huang     1.2               Add sim params
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
 * Config the sandpile param from json file.
 * 
 * 
 * @param p pointer of the param struct
 * @param cfg_path path of the param config path
 * @param sys_log logger of the simulator
 * @return NULL
 * 
 * 
 */
void config_pile_param(PileParam *p, const std::string cfg_path, const logger &sys_log) {
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
        SPDLOG_LOGGER_INFO(sys_log, " -- Used pile type: {}", get_pile_type_name(p->shape));
        SPDLOG_LOGGER_INFO(sys_log, " -- Sandpile width: {}", p->width);
        SPDLOG_LOGGER_INFO(sys_log, " -- Sandpile height: {}", p->height);
        SPDLOG_LOGGER_INFO(sys_log, " -- Initial sand number on center cell: {}", p->ini_sand_num);
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(sys_log, "Parameters read failed. Exception: {}", e.what());
        throw std::runtime_error("Parameters read failed");
    }
    file.close();
}

/**
 * Config the simulation param from json file.
 * 
 * 
 * @param p pointer of the param struct
 * @param cfg_path path of the param config path
 * @param sys_log logger of the simulator
 * @return NULL
 * 
 * 
 */
void config_sim_param(SimParam *p, const std::string cfg_path, const logger &sys_log) {
    json cfg;
    std::ifstream file(cfg_path);
    if (!file.is_open()) {
        SPDLOG_LOGGER_ERROR(sys_log, "Invalid config path!");
        throw std::runtime_error("Open config file failed");
    }
    cfg = json::parse(file);

    try {
        p->max_itr_steps = cfg.at("max_itr_steps");
        SPDLOG_LOGGER_INFO(sys_log, " -- Max simulation steps: {}", p->max_itr_steps);
        if (p->max_itr_steps <= 0) {
            p->max_itr_steps = 20;
            SPDLOG_LOGGER_WARN(sys_log, "Invalid max simulation frame amount, reset to {}", p->max_itr_steps);
        }
        p->data_path = cfg.at("data_path");
        p->video_path = cfg.at("video_path");
        p->outseq_format = (frame_sequence_format)cfg.at("outseq_format");
        p->sp_rate = cfg.at("sp_rate");
        SPDLOG_LOGGER_INFO(sys_log, " -- Output frame sequence format: {}", get_frame_sequence_format_name(p->outseq_format));
        SPDLOG_LOGGER_INFO(sys_log, " -- Output frame sequence sampling gap: {} frame", p->sp_rate);
        if (p->sp_rate < 1) {
            p->sp_rate = 10000;
            SPDLOG_LOGGER_WARN(sys_log, "Invalid output frame sequence sampling gap, reset to {}", p->sp_rate);
        }
        p->bit_rate = cfg.at("bit_rate").get<long long>();
        SPDLOG_LOGGER_INFO(sys_log, " -- Output visualization bit rate: {} (bit/s)", p->bit_rate);
        if (p->bit_rate <= 0LL) {
            p->bit_rate = 100000000LL;
            SPDLOG_LOGGER_WARN(sys_log, "Invalid output visualization bit rate, reset to {} (bit/s)", p->bit_rate);
        }
        p->rc_max_rate = cfg.at("rc_max_rate").get<long long>();
        SPDLOG_LOGGER_INFO(sys_log, " -- Output visualization maximum bitrate: {} (bit/s)", p->rc_max_rate);
        if (p->rc_max_rate <= 0LL) {
            p->rc_max_rate = 100000000LL;
            SPDLOG_LOGGER_WARN(sys_log, "Invalid output visualization maximum bit rate, reset to {} (bit/s)", p->rc_max_rate);
        }
        p->rc_min_rate = cfg.at("rc_min_rate").get<long long>();
        SPDLOG_LOGGER_INFO(sys_log, " -- Output visualization minimum  bitrate: {} (bit/s)", p->rc_min_rate);
        if (p->rc_min_rate <= 0LL) {
            p->rc_min_rate = 100000000LL;
            SPDLOG_LOGGER_WARN(sys_log, "Invalid output visualization minimum  bit rate, reset to {} (bit/s)", p->rc_min_rate);
        }
        p->rc_buffer_size = cfg.at("rc_buffer_size");
        SPDLOG_LOGGER_INFO(sys_log, " -- Output visualization decoder bitstream buffer size: {} (bits)", p->rc_buffer_size);
        if (p->rc_buffer_size <= 0) {
            p->rc_buffer_size = 100000000;
            SPDLOG_LOGGER_WARN(sys_log, "Invalid output visualization decoder bitstream buffer size, reset to {} (bits)", p->rc_buffer_size);
        }
        p->gop_size = cfg.at("gop_size");
        SPDLOG_LOGGER_INFO(sys_log, " -- Output visualization key frame amount: {}", p->gop_size);
        if (p->gop_size < 1) {
            p->gop_size = 10;
            SPDLOG_LOGGER_WARN(sys_log, "Invalid output visualization key frame amount, reset to {}", p->gop_size);
        }
        p->max_b_frames = cfg.at("max_b_frames");
        SPDLOG_LOGGER_INFO(sys_log, " -- Output visualization max bidirectional frame amount: {}", p->max_b_frames);
        if (p->max_b_frames < 0) {
            p->max_b_frames = 0;
            SPDLOG_LOGGER_WARN(sys_log, "Invalid output visualization max bidirectional frame amount, reset to {}", p->max_b_frames);
        }
        p->thread_count = cfg.at("thread_count");
        if (p->thread_count < 0) {
            p->thread_count = 0;
            SPDLOG_LOGGER_WARN(sys_log, "Invalid video codec mthread count, reset to {}", p->thread_count);
        }
    } catch (const std::exception& e) {
        SPDLOG_LOGGER_ERROR(sys_log, "Parameters read failed. Exception: {}", e.what());
        throw std::runtime_error("Parameters read failed");
    }
    file.close();
}

std::map<grid_shape, std::string> PileTypeToString = {
    {grid_shape::TRIANGLE, "TRIANGLE"},
    {grid_shape::QUADRILATERAL, "QUADRILATERAL"},
    {grid_shape::HEXAGON, "HEXAGON"}
};
/**
 * Get the name of enumerate [grid_shape]
 * 
 * 
 * @param input input enumerate instance
 * @return string of the input enumerate instance
 * 
 * 
 */
std::string get_pile_type_name(grid_shape input) {
    return PileTypeToString[input];
}

std::map<frame_sequence_format, std::string> FrameSequenceFormatToString = {
    {frame_sequence_format::SEQUENCE_JPEG, "SEQUENCE_JPEG"},
    {frame_sequence_format::SEQUENCE_PNG, "SEQUENCE_PNG"},
    {frame_sequence_format::SEQUENCE_TIFF, "SEQUENCE_TIFF"},
    {frame_sequence_format::SEQUENCE_BMP, "SEQUENCE_BMP"},
    {frame_sequence_format::SEQUENCE_JPEG2000, "SEQUENCE_JPEG2000"}
};
/**
 * Get the name of enumerate [frame_sequence_format]
 * 
 * 
 * @param input input enumerate instance
 * @return string of the input enumerate instance
 * 
 * 
 */
std::string get_frame_sequence_format_name(frame_sequence_format input) {
    return FrameSequenceFormatToString[input];
}

/**
 * Print the platform information
 * 
 * 
 * @param sys_log logger of the simulator
 * @return NULL
 * 
 * 
 */
void platform_info(const logger &sys_log) {
#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);

    SPDLOG_LOGGER_DEBUG(sys_log, "Platform on Windows:");

    /* Processor arch */
    switch (sysInfo.wProcessorArchitecture) {
    case PROCESSOR_ARCHITECTURE_AMD64:
        SPDLOG_LOGGER_DEBUG(sys_log, " -- Processor arch: x64 (AMD or Intel)");
        break;
    case PROCESSOR_ARCHITECTURE_INTEL:
        SPDLOG_LOGGER_DEBUG(sys_log, " -- Processor arch: x86");
        break;
    case PROCESSOR_ARCHITECTURE_ARM:
        SPDLOG_LOGGER_DEBUG(sys_log, " -- Processor arch: ARM");
        break;
    case PROCESSOR_ARCHITECTURE_ARM64:
        SPDLOG_LOGGER_DEBUG(sys_log, " -- Processor arch: ARM64");
        break;
    case PROCESSOR_ARCHITECTURE_MIPS:
        SPDLOG_LOGGER_DEBUG(sys_log, " -- Processor arch: MIPS");
        break;
    case PROCESSOR_ARCHITECTURE_ALPHA:
        SPDLOG_LOGGER_DEBUG(sys_log, " -- Processor arch: ALPHA");
        break;
    case PROCESSOR_ARCHITECTURE_PPC:
        SPDLOG_LOGGER_DEBUG(sys_log, " -- Processor arch: PowerPC");
        break;
    default:
        SPDLOG_LOGGER_DEBUG(sys_log, " -- Processor arch: Unknown/Other arch type [{}]", sysInfo.wProcessorArchitecture);
        break;
    }
    /* Processor level */
    SPDLOG_LOGGER_DEBUG(sys_log, " -- Processor level: {}", sysInfo.wProcessorLevel);
    /* Processor amount */
    SPDLOG_LOGGER_DEBUG(sys_log, " -- Number of processors: {}", sysInfo.dwNumberOfProcessors);
    /* Processor threads amount */
    DWORD bufferSize = 0;
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer;
    if (!GetLogicalProcessorInformation(nullptr, &bufferSize)) { // get buffer
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            SPDLOG_LOGGER_ERROR(sys_log, "Failed to get buffer size: {}", GetLastError());
            return;
        }
    }
    buffer.resize(bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!GetLogicalProcessorInformation(buffer.data(), &bufferSize)) {
        SPDLOG_LOGGER_ERROR(sys_log, "Failed to get logical processor information: {}", GetLastError());
        return;
    }
    DWORD threadCount = 0;
    for (const auto& info : buffer) {
        if (info.Relationship == RelationProcessorCore) {
            threadCount += (info.ProcessorCore.Flags == LTP_PC_SMT) ? 2 : 1;
        }
    }
    SPDLOG_LOGGER_DEBUG(sys_log, " -- Number of threads: {}", threadCount);
#endif
}

/**
 * Print the start banner of "SANDPILE"
 * 
 * 
 * @param sys_log logger of the simulator
 * @return NULL
 * 
 * 
 */
void print_start_banner(const logger &sys_log) {
    SPDLOG_LOGGER_INFO(sys_log, "███████╗ █████╗ ███╗   ██╗██████╗ ██████╗ ██╗██╗     ███████╗");
    SPDLOG_LOGGER_INFO(sys_log, "██╔════╝██╔══██╗████╗  ██║██╔══██╗██╔══██╗██║██║     ██╔════╝");
    SPDLOG_LOGGER_INFO(sys_log, "███████╗███████║██╔██╗ ██║██║  ██║██████╔╝██║██║     █████╗  ");
    SPDLOG_LOGGER_INFO(sys_log, "╚════██║██╔══██║██║╚██╗██║██║  ██║██╔═══╝ ██║██║     ██╔══╝  ");
    SPDLOG_LOGGER_INFO(sys_log, "███████║██║  ██║██║ ╚████║██████╔╝██║     ██║███████╗███████╗");
    SPDLOG_LOGGER_INFO(sys_log, "╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝     ╚═╝╚══════╝╚══════╝");
}
/**
 * Print the end banner of "FINISHED"
 * 
 * 
 * @param sys_log logger of the simulator
 * @return NULL
 * 
 * 
 */
void print_end_banner(const logger &sys_log) {
    SPDLOG_LOGGER_INFO(sys_log, "███████╗██╗███╗   ██╗██╗███████╗██╗  ██╗███████╗██████╗ ");
    SPDLOG_LOGGER_INFO(sys_log, "██╔════╝██║████╗  ██║██║██╔════╝██║  ██║██╔════╝██╔══██╗");
    SPDLOG_LOGGER_INFO(sys_log, "█████╗  ██║██╔██╗ ██║██║███████╗███████║█████╗  ██║  ██║");
    SPDLOG_LOGGER_INFO(sys_log, "██╔══╝  ██║██║╚██╗██║██║╚════██║██╔══██║██╔══╝  ██║  ██║");
    SPDLOG_LOGGER_INFO(sys_log, "██║     ██║██║ ╚████║██║███████║██║  ██║███████╗██████╔╝");
    SPDLOG_LOGGER_INFO(sys_log, "╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═════╝ ");
}

/**
 * Get the version of SandPile
 * 
 * 
 * @param ver pinter of the SandPileVersion struct
 * @return NULL
 * 
 * 
 */
void get_sandpile_version(SandPileVersion *ver) {
    ver->MajorVersion = SandPileMajorVersion;
    ver->MinorVersion = SandPileMinorVersion;
    ver->UpdateVersion = SandPileUpdateVersion;
}
