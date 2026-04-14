#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "whisper-encoder.h"
#import "whisper-encoder-impl.h"

#import <CoreML/CoreML.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#if __cplusplus
extern "C" {
#endif

// The stock whisper.cpp Core ML encoder is hardcoded to 30s (3000 mel frames).
// Optionally we accept extra shape-specialised variants at 5s / 10s / 15s / 30s,
// loaded as siblings of the requested model. At encode time we pick the smallest
// variant that fits the actual mel length.
#define WHISPER_COREML_MAX_VARIANTS 4

struct whisper_coreml_model_variant {
    int64_t      n_ctx_max; // mel frames the variant was exported for
    const void * data;      // CFBridgingRetain(whisper_encoder_impl)
};

struct whisper_coreml_context {
    int n_variants;
    struct whisper_coreml_model_variant variants[WHISPER_COREML_MAX_VARIANTS];
};

static const char * kVariantSuffixes[WHISPER_COREML_MAX_VARIANTS] = {
    "-5s.mlmodelc",
    "-10s.mlmodelc",
    "-15s.mlmodelc",
    "-30s.mlmodelc",
};

static const int64_t kVariantCtxMax[WHISPER_COREML_MAX_VARIANTS] = {
    500,
    1000,
    1500,
    3000,
};

static const void * whisper_coreml_load_model(NSString * path, MLModelConfiguration * config) {
    NSURL * url = [NSURL fileURLWithPath: path];
    NSError * err = nil;
    whisper_encoder_impl * impl = [[whisper_encoder_impl alloc] initWithContentsOfURL: url configuration: config error: &err];
    if (impl == nil) {
        if (err != nil) {
            NSLog(@"whisper-coreml: failed to load %@: %@", path, err.localizedDescription);
        }
        return NULL;
    }
    return CFBridgingRetain(impl);
}

struct whisper_coreml_context * whisper_coreml_init(const char * path_model) {
    NSString * path_model_str = [[NSString alloc] initWithUTF8String:path_model];

    MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsAll;

    whisper_coreml_context * ctx = new whisper_coreml_context;
    ctx->n_variants = 0;

    // Try shape-specialised variants: <base>-<shape>.mlmodelc next to the requested model.
    NSString * base = path_model_str;
    if ([base hasSuffix: @".mlmodelc"]) {
        base = [base substringToIndex: base.length - (NSUInteger)strlen(".mlmodelc")];
    }

    NSFileManager * fm = [NSFileManager defaultManager];
    for (int i = 0; i < WHISPER_COREML_MAX_VARIANTS; ++i) {
        NSString * candidate = [base stringByAppendingString: [NSString stringWithUTF8String: kVariantSuffixes[i]]];
        if (![fm fileExistsAtPath: candidate]) {
            continue;
        }
        const void * data = whisper_coreml_load_model(candidate, config);
        if (data == NULL) {
            continue;
        }
        ctx->variants[ctx->n_variants].n_ctx_max = kVariantCtxMax[i];
        ctx->variants[ctx->n_variants].data      = data;
        ctx->n_variants++;
    }

    // Fall back to the single stock encoder if no variants were found.
    if (ctx->n_variants == 0) {
        const void * data = whisper_coreml_load_model(path_model_str, config);
        if (data == NULL) {
            delete ctx;
            return NULL;
        }
        ctx->variants[0].n_ctx_max = 3000;
        ctx->variants[0].data      = data;
        ctx->n_variants = 1;
    }

    return ctx;
}

void whisper_coreml_free(struct whisper_coreml_context * ctx) {
    for (int i = 0; i < ctx->n_variants; ++i) {
        if (ctx->variants[i].data != NULL) {
            CFRelease(ctx->variants[i].data);
        }
    }
    delete ctx;
}

// Pick the smallest variant whose shape is >= n_ctx. Variants are sorted
// ascending by n_ctx_max, so a linear scan is fine.
static int whisper_coreml_pick_variant(const whisper_coreml_context * ctx, int64_t n_ctx) {
    for (int i = 0; i < ctx->n_variants; ++i) {
        if (ctx->variants[i].n_ctx_max >= n_ctx) {
            return i;
        }
    }
    return ctx->n_variants - 1; // largest available; caller is responsible for not exceeding it
}

void whisper_coreml_encode(
        const whisper_coreml_context * ctx,
                             int64_t   n_ctx,
                             int64_t   n_mel,
                             int64_t   n_ctx_actual,
                               float * mel,
                               float * out,
                             int64_t   out_nelements) {
    if (n_ctx_actual <= 0 || n_ctx_actual > n_ctx) {
        n_ctx_actual = n_ctx;
    }

    const int idx = whisper_coreml_pick_variant(ctx, n_ctx_actual);
    const int64_t n_ctx_model = ctx->variants[idx].n_ctx_max;

    // When the chosen variant's shape differs from the source buffer stride,
    // repack the valid [n_mel][n_ctx_actual] slice into a dense [n_mel][n_ctx_model]
    // buffer (zero-padded in the time dimension where shorter).
    float * mel_in = mel;
    std::vector<float> padded;
    if (n_ctx_model != n_ctx) {
        padded.assign((size_t)(n_mel * n_ctx_model), 0.0f);
        const int64_t copy_ctx = n_ctx_actual < n_ctx_model ? n_ctx_actual : n_ctx_model;
        for (int64_t m = 0; m < n_mel; ++m) {
            memcpy(padded.data() + m * n_ctx_model,
                   mel + m * n_ctx,
                   (size_t)(copy_ctx * sizeof(float)));
        }
        mel_in = padded.data();
    }

    MLMultiArray * inMultiArray = [
        [MLMultiArray alloc] initWithDataPointer: mel_in
                                           shape: @[@1, @(n_mel), @(n_ctx_model)]
                                        dataType: MLMultiArrayDataTypeFloat32
                                         strides: @[@(n_ctx_model*n_mel), @(n_ctx_model), @1]
                                     deallocator: nil
                                           error: nil
    ];

    @autoreleasepool {
        whisper_encoder_implOutput * outCoreML = [(__bridge id) ctx->variants[idx].data predictionFromLogmel_data:inMultiArray error:nil];

        const size_t written = (size_t) outCoreML.output.count;
        memcpy(out, outCoreML.output.dataPointer, written * sizeof(float));

        if (out_nelements > 0 && (size_t) out_nelements > written) {
            memset(out + written, 0, ((size_t) out_nelements - written) * sizeof(float));
        }
    }
}

#if __cplusplus
}
#endif
