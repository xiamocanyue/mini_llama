#pragma once 

#include <string>
#include <vector>
#include <chrono>

namespace mini_llama {

using RequestClock = std::chrono::steady_clock;

struct TraceEvent {
    std::string stage;
    double use_time_ms = 0.0;
    int use_tokens = 0;
    std::string detail;
};

struct RequestContext {
    void Start();
    void Finish();
    void SetError(const std::string& message);
    void RecordEvent(const std::string& stage, double use_time_ms, int use_tokens = 0, const std::string& detail = "");
    bool IsOk() const;

    std::string trace_id;
    std::string mode;
    std::string backend;
    std::string model_path;

    RequestClock::time_point start_point;
    double total_ms = 0.0;

    int prompt_tokens = 0;
    int prefill_tokens = 0;
    int decode_tokens = 0;
    int generated_tokens = 0;

    double model_load_ms = 0.0;
    double tokenizer_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double sample_ms = 0.0;

    std::string error;
    std::vector<TraceEvent> events;
};

std::string GetNewTraceId();
double GetUseTime(RequestClock::time_point start);
RequestContext StartRequest(const std::string& mode, const std::string& backend, const std::string& model_path);
std::string FormatRequestTraceSummary(const RequestContext& request);
std::vector<std::string> FormatRequestTraceEvents(const RequestContext& request);

}