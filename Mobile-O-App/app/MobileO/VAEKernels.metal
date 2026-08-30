//
// VAEKernels.metal
// Metal compute shaders for VAE post-processing
//

#include <metal_stdlib>
using namespace metal;

/// Normalize VAE decoder output from [-1, 1] to [0, 1].
/// Each thread processes 4 elements via float4 for coalesced memory access.
/// Dispatch: ceil(count / 4) threads in 1D.
kernel void normalizeVAEOutput(
    device const float4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base_idx = gid * 4;
    if (base_idx + 3 < count) {
        float4 v = input[gid];
        output[gid] = clamp(fma(v, 0.5f, 0.5f), 0.0f, 1.0f);
    } else {
        device const float* in_scalar = (device const float*)input;
        device float* out_scalar = (device float*)output;
        for (uint i = base_idx; i < count; i++) {
            out_scalar[i] = clamp(fma(in_scalar[i], 0.5f, 0.5f), 0.0f, 1.0f);
        }
    }
}

/// Fused Float16 → Float32 conversion + [-1, 1] → [0, 1] normalization in a single GPU pass.
/// Eliminates the intermediate FP32 allocation and CPU-side scalar conversion loop.
/// Each thread processes 4 elements via half4 → float4.
/// Dispatch: ceil(count / 4) threads in 1D.
kernel void normalizeVAEOutputFloat16(
    device const half4* input [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base_idx = gid * 4;
    if (base_idx + 3 < count) {
        float4 v = float4(input[gid]);
        output[gid] = clamp(fma(v, 0.5f, 0.5f), 0.0f, 1.0f);
    } else {
        device const half* in_scalar = (device const half*)input;
        device float* out_scalar = (device float*)output;
        for (uint i = base_idx; i < count; i++) {
            float v = float(in_scalar[i]);
            out_scalar[i] = clamp(fma(v, 0.5f, 0.5f), 0.0f, 1.0f);
        }
    }
}

/// Convert a [3, H, W] float tensor in [0, 1] to packed RGBA bytes.
/// Each thread writes one pixel as a single uchar4 store.
/// Dispatch: ceil(W/16) x ceil(H/16) threadgroups of (16, 16).
kernel void floatToRGBA(
    device const float* input [[buffer(0)]],
    device uchar4* output [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& height [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) return;

    uint pixel = gid.y * width + gid.x;
    uint hw = height * width;

    uchar r = uchar(clamp(input[pixel] * 255.0f, 0.0f, 255.0f));
    uchar g = uchar(clamp(input[hw + pixel] * 255.0f, 0.0f, 255.0f));
    uchar b = uchar(clamp(input[2 * hw + pixel] * 255.0f, 0.0f, 255.0f));

    output[pixel] = uchar4(r, g, b, 255);
}

/// Convert packed RGBA bytes to a [1, 3, H, W] float tensor normalized to [0, 1].
/// Inverse of floatToRGBA — used for image editing input preprocessing.
/// Each thread reads one pixel and scatters R, G, B into planar CHW layout.
/// Dispatch: ceil(W/16) x ceil(H/16) threadgroups of (16, 16).
kernel void rgbaToFloatCHW(
    device const uchar4* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& height [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= width || gid.y >= height) return;

    uint pixel = gid.y * width + gid.x;
    uint hw = height * width;
    uchar4 rgba = input[pixel];

    float scale = 1.0f / 255.0f;
    output[pixel]          = float(rgba.r) * scale;
    output[hw + pixel]     = float(rgba.g) * scale;
    output[2 * hw + pixel] = float(rgba.b) * scale;
}
