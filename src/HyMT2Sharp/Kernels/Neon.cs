using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Shared AdvSimd/SDOT helpers for the ARM64 kernels. All entry points assume
/// <see cref="Simd.UseDp"/> (NEON + SDOT); every Apple M-series and recent
/// Cortex/Neoverse qualifies.
/// </summary>
public static unsafe class Neon
{
    /// <summary>Load 8 activation bytes and duplicate into both halves: [a0..a7 | a0..a7].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<sbyte> Dup8(sbyte* p)
    {
        long v = Unsafe.ReadUnaligned<long>(p);
        return Vector128.Create(v, v).AsSByte();
    }

    /// <summary>Broadcast one signed 32-bit group of four activations to all lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<sbyte> Dup4(sbyte* p) =>
        Vector128.Create(Unsafe.ReadUnaligned<int>(p)).AsSByte();

    /// <summary>Load 16 bytes as signed int8.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<sbyte> Load16(sbyte* p) =>
        Unsafe.ReadUnaligned<Vector128<byte>>(p).AsSByte();

    /// <summary>Load 16 bytes as unsigned int8.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<byte> LoadU16(byte* p) =>
        Unsafe.ReadUnaligned<Vector128<byte>>(p);

    /// <summary>acc += a·b, 16 int8 products per instruction (SDOT).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<int> Sdot(Vector128<int> acc, Vector128<sbyte> w, Vector128<sbyte> a) =>
        Dp.DotProduct(acc, w, a);

    /// <summary>acc += a·b treating the weights as unsigned (UDOT).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<int> Udot(Vector128<int> acc, Vector128<byte> w, Vector128<byte> a) =>
        Dp.DotProduct(acc.AsUInt32(), w, a).AsInt32();

    /// <summary>Four-lane horizontal sum.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Reduce(Vector128<int> v) => AdvSimd.Arm64.AddAcross(v).ToScalar();

    /// <summary>Horizontal sum of four f32 lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Reduce(Vector128<float> v) =>
        AdvSimd.Arm64.AddPairwise(AdvSimd.Arm64.AddPairwise(v, v), AdvSimd.Arm64.AddPairwise(v, v)).ToScalar();

    /// <summary>[a0+a1, a2+a3, b0+b1, b2+b3] for int32 lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<int> PairAdd(Vector128<int> a, Vector128<int> b) =>
        AdvSimd.Arm64.AddPairwise(a, b);

    /// <summary>Sign-extend the low four s8 lanes to i32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<int> WidenLo4(Vector64<sbyte> v) =>
        AdvSimd.SignExtendWideningLower(AdvSimd.SignExtendWideningLower(v).GetLower());

    /// <summary>Load four consecutive s8 scales as an i32 vector.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<int> LoadScales4(sbyte* p) =>
        WidenLo4(Vector128.CreateScalar(Unsafe.ReadUnaligned<int>(p)).AsSByte().GetLower());

    /// <summary>Sum of 16 int16 bsums, widened to i32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int BsumAll(short* bsums)
    {
        Vector128<short> s = AdvSimd.Add(
            Unsafe.ReadUnaligned<Vector128<short>>(bsums),
            Unsafe.ReadUnaligned<Vector128<short>>(bsums + 8));
        return AdvSimd.Arm64.AddAcrossWidening(s).ToScalar();
    }

    /// <summary>Sum of the 16 int16 bsums of one BlockQ8K, as int.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int BsumAll(BlockQ8K* y) => BsumAll(y->Bsums);
}
