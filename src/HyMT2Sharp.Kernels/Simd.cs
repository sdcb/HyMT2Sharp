using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// ISA dispatch order: AVX-VNNI → AVX2 → AdvSimd+SDOT (ARM64) → portable
/// <see cref="System.Numerics.Vector{T}"/> → scalar (kept for tests/reference only).
/// Setting HYMT2SHARP_FORCE_PORTABLE=1 skips the hardware paths so the
/// Vector&lt;T&gt; tier can be exercised everywhere; on ARM64 it is the tier
/// below AdvSimd rather than the only option.
/// </summary>
public static class Simd
{
    public static readonly bool ForcePortable =
        Environment.GetEnvironmentVariable("HYMT2SHARP_FORCE_PORTABLE") is "1" or "true" or "TRUE";

    public static bool UseAvx2 => Avx2.IsSupported && !ForcePortable;
    public static bool UseAvxVnni => AvxVnni.IsSupported && !ForcePortable;
    public static bool UseAvx => Avx.IsSupported && !ForcePortable;
    public static bool UseFma => Fma.IsSupported && !ForcePortable;

    /// <summary>128-bit NEON (mandatory on ARM64; also gates TBL lookups).</summary>
    public static bool UseAdvSimd => AdvSimd.Arm64.IsSupported && !ForcePortable;

    /// <summary>ARM64 dot-product extension (SDOT/UDOT): 16 int8 MACs per instruction.</summary>
    public static bool UseDp => Dp.Arm64.IsSupported && !ForcePortable;

    /// <summary>
    /// Whether the x8 packed weight panels have a fast kernel on this machine
    /// (AVX2 on x86, SDOT on ARM64). Repack runs once at load, so pack whenever
    /// either kernel can consume them.
    /// </summary>
    public static bool UsePanels => UseAvx2 || UseDp;
}
