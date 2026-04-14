// Wrapper of the Core ML Whisper Encoder model
//
// Code is derived from the work of Github user @wangchou
// ref: https://github.com/wangchou/callCoreMLFromCpp

#include <stdint.h>

#if __cplusplus
extern "C" {
#endif

struct whisper_coreml_context;

// Loads a Core ML encoder for the given base path (".mlmodelc" appended).
//
// If sibling files `<base>-5s.mlmodelc`, `<base>-10s.mlmodelc`, `<base>-15s.mlmodelc`,
// `<base>-30s.mlmodelc` exist next to the requested model, they are all loaded as
// shape-specialised variants and dispatched at encode time based on the actual
// mel length. When no variants are found, behavior is unchanged — a single
// model is loaded at the exact path passed in.
struct whisper_coreml_context * whisper_coreml_init(const char * path_model);
void whisper_coreml_free(struct whisper_coreml_context * ctx);

// Runs the encoder.
//
// Parameters:
//   n_ctx        — time dimension (stride) of the source mel buffer
//   n_mel        — mel bin count
//   n_ctx_actual — valid mel frames in the source buffer. When the context has
//                  multiple shape variants, the smallest variant that fits
//                  `n_ctx_actual` is selected. Pass the same value as `n_ctx`
//                  to preserve stock (full-pad) behavior.
//   mel          — mel data laid out as [n_mel][n_ctx]
//   out          — output buffer
//   out_nelements— total float capacity of `out`. When the selected variant
//                  produces fewer elements, the tail of `out` is zeroed so
//                  downstream tensors of fixed size remain well-defined.
//   out_n_ctx_enc— if non-null, receives the number of audio-ctx tokens
//                  actually produced by the selected variant (i.e. the
//                  variant's post-conv2 time dimension). The caller can use
//                  this to restrict downstream cross-attention — positions
//                  beyond this range contain zeros, not encoder output, and
//                  attending to them would pollute attention with bias-only
//                  keys/values.
void whisper_coreml_encode(
        const whisper_coreml_context * ctx,
                             int64_t   n_ctx,
                             int64_t   n_mel,
                             int64_t   n_ctx_actual,
                               float * mel,
                               float * out,
                             int64_t   out_nelements,
                             int64_t * out_n_ctx_enc);

#if __cplusplus
}
#endif
