// `qubivex_compute::math::exp` for the Vulkan shaders: the same operations in the same
// order (basic operations and fma only), so with contraction off it has the CPU's bits.
// Shared by swiglu.comp and route.comp. A shader that includes it has exp's rounding
// idiom `(x * LOG2E + SHIFT) - SHIFT`, so it must be built -O0 (build.sh; audit §15.1).

// `qubivex_compute::math` constants, as bit patterns.
const float LOG2E = uintBitsToFloat(0x3fb8aa3bu);
const float LN2_HI = uintBitsToFloat(0x3f317200u);
const float LN2_LO = uintBitsToFloat(0x35bfbe8eu);
const float SHIFT = uintBitsToFloat(0x4b400000u);
const float OVERFLOW = uintBitsToFloat(0x42b17218u);
const float UNDERFLOW = uintBitsToFloat(0xc2aeac50u);
const float MIN_POSITIVE = uintBitsToFloat(0x00800000u);
const float INF = uintBitsToFloat(0x7f800000u);

float pow2(int k) {
    return intBitsToFloat((k + 127) << 23);
}

float qexp(float x) {
    if (isnan(x)) {
        return x;
    }
    if (x > OVERFLOW) {
        return INF;
    }
    if (x < UNDERFLOW) {
        return 0.0;
    }
    float kf = (x * LOG2E + SHIFT) - SHIFT;
    float r = fma(-kf, LN2_HI, x);
    r = fma(-kf, LN2_LO, r);
    float q = uintBitsToFloat(0x39506967u);
    q = fma(q, r, uintBitsToFloat(0x3ab743ceu));
    q = fma(q, r, uintBitsToFloat(0x3c088908u));
    q = fma(q, r, uintBitsToFloat(0x3d2aa9c1u));
    q = fma(q, r, uintBitsToFloat(0x3e2aaaaau));
    q = fma(q, r, uintBitsToFloat(0x3f000000u));
    float y = fma(q, r * r, r) + 1.0;
    int k = int(kf);
    int k1 = k >> 1;
    float s = (y * pow2(k1)) * pow2(k - k1);
    return s < MIN_POSITIVE ? 0.0 : s;
}
