#include "include/request_context.h"


#include <atomic>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace mini_llama {
namespace {

std::atomic<unsigned long long> g_request_counter{0};

std::string FormatMs(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << value;
    return out.str();
}

}//namespace

void RequestContext::Start() {
    start_point = RequestClock::now();
}

void RequestContext::Finish() {
    total_ms = GetUseTime(start_point);
}

void RequestContext::SetError(const std::string& message) {
    error = message;
}

void RequestContext::RecordEvent(const std::string& stage, double use_time_ms, 
                                int use_tokens, const std::string& detail) {
    events.push_back({stage, use_time_ms, use_tokens, detail});
}

bool RequestContext::IsOk() const {
    return error.empty();
}

std::string GetNewTraceId() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    unsigned long long seq = g_request_counter++;
    std::tm* tm = std::localtime(&time);
    std::ostringstream out;
    out << "req_" << std::put_time(tm, "%Y%m%d%H%M%S") << "_" 
        << std::setw(6) << std::setfill('0') << seq;

    return out.str();
}

double GetUseTime(RequestClock::time_point start) {
    return std::chrono::duration<double, std::milli>(RequestClock::now() - start).count();
}

RequestContext StartRequest(const std::string& mode, const std::string& backend, 
                            const std::string& model_path) {
    RequestContext request;
    request.mode = mode;
    request.backend = backend;
    request.model_path = model_path;
    request.trace_id = GetNewTraceId();
    request.Start();
    return request;
}

std::string FormatRequestTraceSummary(const RequestContext& request) {
    std::ostringstream out;
    out << "trace summary: " << "\n"
        << " trace_id=" << request.trace_id
        << "  mode=" << request.mode
        << "  status=" << (request.IsOk() ? "OK" : "error")
        << "  backend=" << request.backend
        << "  model=" << request.model_path
        << "  prompt_tokens=" << request.prompt_tokens
        << "  prefill_tokens=" << request.prefill_tokens
        << "  decode_tokens=" << request.decode_tokens
        << "  generated_tokens=" << request.generated_tokens
        << "  model_load_ms=" << FormatMs(request.model_load_ms)
        << "  tokenizer_ms=" << FormatMs(request.tokenizer_ms)
        << "  prefill_ms=" << FormatMs(request.prefill_ms)
        << "  decode_ms=" << FormatMs(request.decode_ms)
        << "  sample_ms=" << FormatMs(request.sample_ms)
        << "  total_ms=" << FormatMs(request.total_ms);
    if (!request.IsOk()) {
        out << " error=\"" << request.error << "\"";
    }
    return out.str();
}

std::vector<std::string> FormatRequestTraceEvents(const RequestContext& request) {
    std::vector<std::string> lines;
    for (const TraceEvent& event : request.events) {
        std::ostringstream out;
        out << "trace event: trace_id=" << request.trace_id
            << "  stage=" << event.stage
            << "  use_tokens=" << event.use_tokens
            << "  use_time_ms=" << event.use_time_ms;
        if (!event.detail.empty()) {
            out << "\"" << event.detail << "\"";
        }
        lines.push_back(out.str());
    }
    return lines;
}

}