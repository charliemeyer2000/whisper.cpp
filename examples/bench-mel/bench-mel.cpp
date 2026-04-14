// bench-mel — micro-benchmark for whisper_pcm_to_mel (log-mel spectrogram).
//
// Times only the mel computation (not the encoder), so candidate
// optimizations to the log_mel_spectrogram path can be validated in
// isolation. To compare two implementations, build each ref, run with a
// distinct --label, and diff the JSON summaries:
//
//     git checkout baseline  && cmake --build build
//     ./build/bin/whisper-bench-mel -m model.bin -f jfk.wav -l baseline  > baseline.json
//     git checkout candidate && cmake --build build
//     ./build/bin/whisper-bench-mel -m model.bin -f jfk.wav -l candidate > candidate.json
//
// Reports min / median / mean / p95 / p99 across N iterations, after a
// configurable warm-up. Emits a JSON object on stdout for artifact
// capture; human-readable summary on stderr.

#include "common-whisper.h"
#include "whisper.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct bench_params {
    std::string model;
    std::string audio;
    int32_t     n_threads    = std::max(1, std::min(4, (int32_t) std::thread::hardware_concurrency()));
    int32_t     n_iterations = 100;
    int32_t     n_warmup     = 5;
    std::string label;               // free-form tag echoed into the JSON summary
};

// Minimal JSON string escape so free-form labels don't produce invalid JSON.
// Stricter than RFC 8259 allows (escapes all <0x20), but simplicity beats
// completeness for a benchmark tool.
static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char) c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char) c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static void print_usage(const char * argv0, const bench_params & p) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv0);
    fprintf(stderr, "\n");
    fprintf(stderr, "  -m FNAME,  --model FNAME      [required] whisper model path (mel filter weights)\n");
    fprintf(stderr, "  -f FNAME,  --file FNAME       [required] input audio (WAV/FLAC/mp3 via miniaudio)\n");
    fprintf(stderr, "  -n N,      --iterations N     [%-7d] timed iterations after warmup\n", p.n_iterations);
    fprintf(stderr, "  -w N,      --warmup N         [%-7d] untimed warmup iterations\n", p.n_warmup);
    fprintf(stderr, "  -t N,      --threads N        [%-7d] threads passed to whisper_pcm_to_mel\n", p.n_threads);
    fprintf(stderr, "  -l TAG,    --label TAG        [%-7s] free-form tag echoed into JSON summary\n", p.label.c_str());
    fprintf(stderr, "  -h,        --help                     show this help\n");
    fprintf(stderr, "\n");
}

static bool parse_args(int argc, char ** argv, bench_params & p) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s requires a value\n", flag); std::exit(1); }
            return argv[++i];
        };
        if      (a == "-h" || a == "--help")       { print_usage(argv[0], p); std::exit(0); }
        else if (a == "-m" || a == "--model")      { p.model        = next("-m"); }
        else if (a == "-f" || a == "--file")       { p.audio        = next("-f"); }
        else if (a == "-n" || a == "--iterations") { p.n_iterations = std::atoi(next("-n")); }
        else if (a == "-w" || a == "--warmup")     { p.n_warmup     = std::atoi(next("-w")); }
        else if (a == "-t" || a == "--threads")    { p.n_threads    = std::atoi(next("-t")); }
        else if (a == "-l" || a == "--label")      { p.label        = next("-l"); }
        else { fprintf(stderr, "error: unknown arg: %s\n", a.c_str()); print_usage(argv[0], p); return false; }
    }
    if (p.model.empty() || p.audio.empty()) {
        fprintf(stderr, "error: -m and -f are required\n");
        print_usage(argv[0], p);
        return false;
    }
    if (p.n_iterations <= 0) {
        fprintf(stderr, "error: --iterations must be > 0 (got %d)\n", p.n_iterations);
        return false;
    }
    if (p.n_warmup < 0) {
        fprintf(stderr, "error: --warmup must be >= 0 (got %d)\n", p.n_warmup);
        return false;
    }
    if (p.n_threads <= 0) {
        fprintf(stderr, "error: --threads must be > 0 (got %d)\n", p.n_threads);
        return false;
    }
    return true;
}

static double percentile(const std::vector<double> & sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double idx = q * (double) (sorted.size() - 1);
    const size_t lo = (size_t) idx;
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = idx - (double) lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

int main(int argc, char ** argv) {
    bench_params params;
    if (!parse_args(argc, argv, params)) return 1;

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(params.audio, pcmf32, pcmf32s, /*stereo=*/false)) {
        fprintf(stderr, "error: failed to read audio: %s\n", params.audio.c_str());
        return 1;
    }
    const double audio_s = (double) pcmf32.size() / (double) WHISPER_SAMPLE_RATE;

    // log_mel_spectrogram runs entirely on the CPU regardless of use_gpu, and
    // whisper_pcm_to_mel never touches any ggml backend. Disable GPU init so
    // the bench doesn't carry backend setup cost it won't use. This also
    // sidesteps ggml_backend_load_all() — unnecessary for mel, required for
    // encode/decode on some platforms.
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to load model: %s\n", params.model.c_str());
        return 1;
    }

    fprintf(stderr, "bench-mel: model=%s audio=%s (%.3fs, %zu samples)\n",
            params.model.c_str(), params.audio.c_str(), audio_s, pcmf32.size());
    fprintf(stderr, "bench-mel: threads=%d iterations=%d warmup=%d label=%s\n",
            params.n_threads, params.n_iterations, params.n_warmup, params.label.c_str());

    for (int i = 0; i < params.n_warmup; ++i) {
        if (whisper_pcm_to_mel(ctx, pcmf32.data(), (int) pcmf32.size(), params.n_threads) != 0) {
            fprintf(stderr, "error: whisper_pcm_to_mel failed during warmup\n");
            whisper_free(ctx);
            return 1;
        }
    }

    std::vector<double> timings_ms;
    timings_ms.reserve(params.n_iterations);
    for (int i = 0; i < params.n_iterations; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (whisper_pcm_to_mel(ctx, pcmf32.data(), (int) pcmf32.size(), params.n_threads) != 0) {
            fprintf(stderr, "error: whisper_pcm_to_mel failed at iter %d\n", i);
            whisper_free(ctx);
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        timings_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::vector<double> sorted = timings_ms;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (double t : timings_ms) sum += t;
    const double mean   = sum / (double) timings_ms.size();
    const double p_min  = sorted.front();
    const double p_max  = sorted.back();
    const double p_50   = percentile(sorted, 0.50);
    const double p_95   = percentile(sorted, 0.95);
    const double p_99   = percentile(sorted, 0.99);

    fprintf(stderr, "\nbench-mel results (ms, over %d iterations):\n", (int) timings_ms.size());
    fprintf(stderr, "  min     = %8.3f\n", p_min);
    fprintf(stderr, "  median  = %8.3f\n", p_50);
    fprintf(stderr, "  mean    = %8.3f\n", mean);
    fprintf(stderr, "  p95     = %8.3f\n", p_95);
    fprintf(stderr, "  p99     = %8.3f\n", p_99);
    fprintf(stderr, "  max     = %8.3f\n", p_max);
    fprintf(stderr, "  xRT     = %8.3f  (audio %.3fs / median %.3fms)\n",
            (audio_s * 1000.0) / p_50, audio_s, p_50);

    fprintf(stdout,
        "{\"label\":\"%s\",\"audio_s\":%.6f,\"samples\":%zu,"
        "\"threads\":%d,\"iterations\":%d,\"warmup\":%d,"
        "\"min_ms\":%.6f,\"median_ms\":%.6f,\"mean_ms\":%.6f,"
        "\"p95_ms\":%.6f,\"p99_ms\":%.6f,\"max_ms\":%.6f}\n",
        json_escape(params.label).c_str(),
        audio_s, pcmf32.size(),
        params.n_threads, params.n_iterations, params.n_warmup,
        p_min, p_50, mean, p_95, p_99, p_max);

    whisper_free(ctx);
    return 0;
}
