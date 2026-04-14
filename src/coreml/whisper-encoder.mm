#if !__has_feature(objc_arc)
#error This file must be compiled with automatic reference counting enabled (-fobjc-arc)
#endif

#import "whisper-encoder.h"
#import "whisper-encoder-impl.h"

#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>

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
        NSLog(@"whisper-coreml: loaded single encoder at %@", path_model_str);
    } else {
        NSMutableArray<NSString *> * shapes = [NSMutableArray array];
        for (int i = 0; i < ctx->n_variants; ++i) {
            [shapes addObject: [NSString stringWithFormat: @"%lldctx", (long long) ctx->variants[i].n_ctx_max]];
        }
        NSLog(@"whisper-coreml: loaded %d shape variant(s): %@", ctx->n_variants, [shapes componentsJoinedByString: @","]);
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

int64_t whisper_coreml_n_ctx_enc_for(const whisper_coreml_context * ctx, int64_t n_ctx_actual) {
    if (ctx == NULL || ctx->n_variants <= 0) {
        return 0;
    }
    const int idx = whisper_coreml_pick_variant(ctx, n_ctx_actual);
    return ctx->variants[idx].n_ctx_max / 2;
}

void whisper_coreml_encode(
        const whisper_coreml_context * ctx,
                             int64_t   n_ctx,
                             int64_t   n_mel,
                             int64_t   n_ctx_actual,
                               float * mel,
                               float * out,
                             int64_t   out_nelements,
                             int64_t * out_n_ctx_enc) {
    if (n_ctx_actual <= 0 || n_ctx_actual > n_ctx) {
        n_ctx_actual = n_ctx;
    }

    const int idx = whisper_coreml_pick_variant(ctx, n_ctx_actual);
    const int64_t n_ctx_model = ctx->variants[idx].n_ctx_max;
    // The encoder stem has a stride-2 conv2; the audio-ctx (output time dim)
    // is always n_ctx_model / 2. This is what downstream cross-attention
    // should see as the valid range.
    const int64_t n_ctx_enc = n_ctx_model / 2;

    if (out_n_ctx_enc != NULL) {
        *out_n_ctx_enc = n_ctx_enc;
    }

    if (ctx->n_variants > 1) {
        NSLog(@"whisper-coreml: encode n_ctx_actual=%lld picked variant=%lldctx n_ctx_enc=%lld",
              (long long) n_ctx_actual, (long long) n_ctx_model, (long long) n_ctx_enc);
    }

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
        NSError * pred_err = nil;
        whisper_encoder_implOutput * outCoreML = [(__bridge id) ctx->variants[idx].data predictionFromLogmel_data:inMultiArray error:&pred_err];

        if (outCoreML == nil || outCoreML.output == nil) {
            NSLog(@"whisper-coreml: predictionFromLogmel_data failed: %@",
                  pred_err ? pred_err.localizedDescription : @"(no error info)");
            if (out_nelements > 0) {
                memset(out, 0, (size_t) out_nelements * sizeof(float));
            }
        } else {
            const size_t produced = (size_t) outCoreML.output.count;
            // Clamp against destination capacity: the caller sizes `out`
            // from the variant it expects (via whisper_coreml_n_ctx_enc_for),
            // but guard against a malformed model producing more elements.
            const size_t cap = out_nelements > 0 ? (size_t) out_nelements : produced;
            const size_t written = produced < cap ? produced : cap;
            if (written > 0 && outCoreML.output.dataPointer != NULL) {
                // ANE-targeted mlprograms (compute_precision=FLOAT16) emit FP16
                // outputs. Convert to FP32 on the way into the ggml embd_enc
                // buffer; fall back to a straight memcpy for FP32 outputs
                // (stock 3000-ctx encoder, historical builds).
                const MLMultiArrayDataType dtype = outCoreML.output.dataType;
                if (dtype == MLMultiArrayDataTypeFloat16) {
                    vImage_Buffer src = {
                        .data = outCoreML.output.dataPointer,
                        .height = 1,
                        .width = written,
                        .rowBytes = written * sizeof(uint16_t),
                    };
                    vImage_Buffer dst = {
                        .data = out,
                        .height = 1,
                        .width = written,
                        .rowBytes = written * sizeof(float),
                    };
                    vImageConvert_Planar16FtoPlanarF(&src, &dst, 0);
                } else {
                    memcpy(out, outCoreML.output.dataPointer, written * sizeof(float));
                }
            }
            if (out_nelements > 0 && (size_t) out_nelements > written) {
                memset(out + written, 0, ((size_t) out_nelements - written) * sizeof(float));
            }
        }
    }
}

#if __cplusplus
}
#endif
