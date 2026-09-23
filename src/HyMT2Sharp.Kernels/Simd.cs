using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// ISA dispatch order: AVX-VNNI → AVX2 → portable <see cref="System.Numerics.Vector{T}"/>
/// → scalar (kept for tests/reference only). Setting HYMT2SHARP_FORCE_PORTABLE=1 skips
/// the x86 paths so the Vector&lt;T&gt; tier can be exercised on AVX2 hardware; on ARM64
/// it is reached naturally because the x86 properties report false.
/// </summary>
public static class Simd
{
    public static readonly bool ForcePortable =
        Environment.GetEnvironmentVariable("HYMT2SHARP_FORCE_PORTABLE") is "1" or "true" or "TRUE";

    public static bool UseAvx2 => Avx2.IsSupported && !ForcePortable;
    public static bool UseAvxVnni => AvxVnni.IsSupported && !ForcePortable;
    public static bool UseAvx => Avx.IsSupported && !ForcePortable;
    public static bool UseFma => Fma.IsSupported && !ForcePortable;
}
