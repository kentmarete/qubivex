// `qubivex_compute::math::rms_scale`'s last step for the Vulkan shaders:
// `1 / sqrt(ss / n + eps)` from the sum of squares `ss`, the length `n` (as f32) and
// `eps` (as f32), with the CPU's operations in its order: a division, an add, a square
// root and a division, each rounded once.
//
// Shared by resnorm.comp and rms_scale.comp (the conformance probe), both built -O0, so
// no optimiser rewrites `1 / sqrt(q)` as `inversesqrt(q)` (which is not correctly
// rounded). The add is NoContraction (build.sh); there is no multiply to fuse into it.
// The divisions and the square root must be correctly rounded: the exhaustive
// conformance test checks every f32 input on the device (audit §18).

float rms_scale_of(float ss, float n, float eps) {
    float q = ss / n + eps;
    return 1.0 / sqrt(q);
}
