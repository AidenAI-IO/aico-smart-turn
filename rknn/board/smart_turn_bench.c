/*
 * Benchmark and verify the Smart Turn RKNN model on an RV1103/RV1106 board.
 *
 *   smart_turn_bench MODEL.rknn [--iters N]
 *   smart_turn_bench MODEL.rknn --features FILE|- --count K [--iters N]
 *   smart_turn_bench MODEL.rknn --audio FILE|- --count K [--iters N]
 *
 * Default: N timed runs of the full path (log-mel features in C, int8
 * quantization, rknn_run) on synthetic audio, plus memory usage.
 *
 * --features: K float32 inputs of the model's input shape, e.g. (1, 80, 400),
 *     concatenated ("-" for stdin). Prints "PROB <i> <p>" per input.
 * --audio: K records of [u32 n][int16 n] 16 kHz mono PCM ("-" for stdin), run
 *     through the full path. Prints "PROB <i> <p> <features_ms> <run_ms>".
 *
 * The RV1106 runtime (librknnmrt) does not implement RKNN_QUERY_PERF_*; set
 * RKNN_LOG_LEVEL=5 in the environment to get per-layer timings instead.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "smart_turn.h"

static double now_ms(clockid_t clock) {
    struct timespec ts;
    clock_gettime(clock, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static long proc_kb(const char *path, const char *key) {
    FILE *f = fopen(path, "r");
    char line[256];
    long value = -1;
    size_t n = strlen(key);
    if (!f) return -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, n) == 0 && line[n] == ':') {
            value = strtol(line + n + 1, NULL, 10);
            break;
        }
    }
    fclose(f);
    return value;
}

static void print_memory(const char *stage) {
    printf("MEM %-12s VmRSS=%ld kB VmHWM=%ld kB MemAvailable=%ld kB\n", stage,
           proc_kb("/proc/self/status", "VmRSS"), proc_kb("/proc/self/status", "VmHWM"),
           proc_kb("/proc/meminfo", "MemAvailable"));
}

static void print_attr(const char *kind, const rknn_tensor_attr *a) {
    printf("%s %s dims=[", kind, a->name);
    for (uint32_t i = 0; i < a->n_dims; i++) printf("%s%u", i ? "," : "", a->dims[i]);
    printf("] n_elems=%u size_with_stride=%u fmt=%s type=%s qnt=%s zp=%d scale=%g\n", a->n_elems,
           a->size_with_stride, get_format_string(a->fmt), get_type_string(a->type),
           get_qnt_type_string(a->qnt_type), a->zp, a->scale);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n) {
    qsort(v, n, sizeof *v, cmp_double);
    return v[n / 2];
}

static FILE *open_input(const char *path) { return strcmp(path, "-") ? fopen(path, "rb") : stdin; }

int main(int argc, char **argv) {
    const char *model_path = NULL, *features_path = NULL, *audio_path = NULL;
    int iters = 50, count = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--features") && i + 1 < argc) features_path = argv[++i];
        else if (!strcmp(argv[i], "--audio") && i + 1 < argc) audio_path = argv[++i];
        else if (!strcmp(argv[i], "--count") && i + 1 < argc) count = atoi(argv[++i]);
        else model_path = argv[i];
    }
    if (!model_path || iters < 1 || ((features_path || audio_path) && count < 1)) {
        fprintf(stderr, "usage: %s MODEL.rknn [--iters N] [--features|--audio FILE|- --count K]\n", argv[0]);
        return 2;
    }

    print_memory("start");
    char err[256];
    double t0 = now_ms(CLOCK_MONOTONIC);
    smart_turn *st = smart_turn_open(model_path, err, sizeof err);
    if (!st) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    printf("TIME init=%.1f ms\n", now_ms(CLOCK_MONOTONIC) - t0);

    rknn_context ctx = smart_turn_rknn(st);
    rknn_sdk_version ver;
    if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof ver) == RKNN_SUCC)
        printf("VERSION api=%s driver=%s\n", ver.api_version, ver.drv_version);
    rknn_mem_size mem;
    if (rknn_query(ctx, RKNN_QUERY_MEM_SIZE, &mem, sizeof mem) == RKNN_SUCC)
        printf("NPU_MEM weight=%u kB internal=%u kB dma_total=%llu kB\n", mem.total_weight_size / 1024,
               mem.total_internal_size / 1024, (unsigned long long)mem.total_dma_allocated_size / 1024);
    print_attr("INPUT ", smart_turn_input_attr(st));
    print_attr("OUTPUT", smart_turn_output_attr(st));
    print_memory("after_init");

    const int window = smart_turn_window_samples(st), n_features = smart_turn_feature_count(st);
    int16_t *pcm = malloc(window * sizeof(int16_t));
    float *features = malloc(n_features * sizeof(float));
    float prob = 0;
    smart_turn_timing t;

    if (features_path) {
        FILE *f = open_input(features_path);
        for (int i = 0; f && i < count; i++) {
            if (fread(features, sizeof(float), n_features, f) != (size_t)n_features) {
                fprintf(stderr, "short read at sample %d\n", i);
                return 1;
            }
            if (smart_turn_predict_features(st, features, &prob, &t)) return 1;
            printf("PROB %d %.6f\n", i, prob);
        }
        if (f && f != stdin) fclose(f);
    } else if (audio_path) {
        FILE *f = open_input(audio_path);
        size_t capacity = 0;
        int16_t *turn = NULL;
        for (int i = 0; f && i < count; i++) {
            uint32_t n;
            if (fread(&n, sizeof n, 1, f) != 1) {
                fprintf(stderr, "short read at sample %d\n", i);
                return 1;
            }
            if (n > capacity) turn = realloc(turn, (capacity = n) * sizeof(int16_t));
            if (fread(turn, sizeof(int16_t), n, f) != n) {
                fprintf(stderr, "short read at sample %d\n", i);
                return 1;
            }
            if (smart_turn_predict(st, turn, n, &prob, &t)) return 1;
            printf("PROB %d %.6f %.2f %.2f\n", i, prob, t.features_ms, t.run_ms);
        }
        free(turn);
        if (f && f != stdin) fclose(f);
    }

    /* Latency of the full path on synthetic speech-like audio. */
    for (int i = 0; i < window; i++)
        pcm[i] = (int16_t)(8000 * sinf(i * 0.05f) * sinf(i * 0.0007f) + (rand() % 400) - 200);
    double *feat = malloc(iters * sizeof(double)), *quant = malloc(iters * sizeof(double)),
           *run = malloc(iters * sizeof(double)), *total = malloc(iters * sizeof(double));
    double cpu_total = 0;
    for (int i = 0; i < iters + 3; i++) {
        double w0 = now_ms(CLOCK_MONOTONIC), c0 = now_ms(CLOCK_PROCESS_CPUTIME_ID);
        if (smart_turn_predict(st, pcm, window, &prob, &t)) return 1;
        double w1 = now_ms(CLOCK_MONOTONIC), c1 = now_ms(CLOCK_PROCESS_CPUTIME_ID);
        if (i < 3) continue; /* warm-up */
        feat[i - 3] = t.features_ms;
        quant[i - 3] = t.quantize_ms;
        run[i - 3] = t.run_ms;
        total[i - 3] = w1 - w0;
        cpu_total += c1 - c0;
    }
    printf("LATENCY iters=%d median: features=%.2f ms quantize=%.2f ms rknn_run=%.2f ms total=%.2f ms; "
           "process CPU time avg=%.2f ms; last prob=%.4f\n",
           iters, median(feat, iters), median(quant, iters), median(run, iters), median(total, iters),
           cpu_total / iters, prob);
    print_memory("end");

    smart_turn_close(st);
    free(pcm);
    free(features);
    free(feat);
    free(quant);
    free(run);
    free(total);
    return 0;
}
