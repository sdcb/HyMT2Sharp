using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// 4×8 Q8_0 panel GEMM. One activation dword broadcasts across 8 columns.
/// AVX-VNNI uses <c>vpdpbusd</c> on weights biased by +128. AVX2 uses the sign trick so
/// <c>vpmaddubsw</c> stays inside int16.
/// </summary>
public static unsafe class GemmQ8_0
{
    public static void Gemm8x8(int n, float* dst, int ldc, BlockQ8_0x8* weights, BlockQ8_0x4* activations, int rows, int cols)
    {
        if ((rows & 3) != 0)
            throw new ArgumentException("rows must be a multiple of 4.", nameof(rows));
        if ((cols & 7) != 0)
            throw new ArgumentException("cols must be a multiple of 8.", nameof(cols));
        if (AvxVnni.IsSupported)
            GemmVnni(n, dst, ldc, weights, activations, rows, cols);
        else if (Avx2.IsSupported)
            GemmAvx2(n, dst, ldc, weights, activations, rows, cols);
        else
            GemmScalar(n, dst, ldc, weights, activations, rows, cols);
    }

    public static void GemmScalar(int n, float* dst, int ldc, BlockQ8_0x8* weights, BlockQ8_0x4* activations, int rows, int cols)
    {
        int nb = n / Qk.Q8_0Block;
        for (int y = 0; y < rows / 4; y++)
        {
            BlockQ8_0x4* aPtr = activations + y * nb;
            for (int x = 0; x < cols / 8; x++)
            {
                BlockQ8_0x8* bPtr = weights + x * nb;
                for (int r = 0; r < 4; r++)
                {
                    for (int c = 0; c < 8; c++)
                    {
                        float sum = 0;
                        for (int b = 0; b < nb; b++)
                        {
                            int acc = 0;
                            for (int k = 0; k < Qk.Q8_0Block; k++)
                            {
                                int w = (sbyte)bPtr[b].Qs[(k >> 2) * 32 + c * 4 + (k & 3)];
                                int a = aPtr[b].Qs[r * Qk.Q8_0Block + k];
                                acc += w * a;
                            }

                            sum += bPtr[b].D[c] * aPtr[b].D[r] * acc;
                        }

                        dst[(y * 4 + r) * ldc + x * 8 + c] = sum;
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmAvx2(int n, float* dst, int ldc, BlockQ8_0x8* weights, BlockQ8_0x4* activations, int rows, int cols) =>
        GemmTiles(n, dst, ldc, weights, activations, rows, cols, vnni: false);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmVnni(int n, float* dst, int ldc, BlockQ8_0x8* weights, BlockQ8_0x4* activations, int rows, int cols) =>
        GemmTiles(n, dst, ldc, weights, activations, rows, cols, vnni: true);

    /// <summary>
    /// Eight columns × four K. <paramref name="w"/> is signed int8 bit patterns,
    /// <paramref name="act"/> points at four signed activation bytes.
    /// </summary>
    /// <summary>
    /// Eight columns × four K. Activations must be in −127..127 so the sign trick
    /// does not overflow the int16 <c>vpmaddubsw</c> pair sum. The quantizer clamps to that range.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<int> Dot4Avx2(Vector256<byte> w, int* act)
    {
        Vector256<sbyte> ws = w.AsSByte();
        Vector256<sbyte> a = Avx2.BroadcastScalarToVector256(act).AsSByte();
        Vector256<byte> aw = Avx2.Sign(ws, ws).AsByte();
        Vector256<sbyte> sa = Avx2.Sign(a, ws);
        return Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(aw, sa), Vector256<short>.One);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<int> Dot4Vnni(Vector256<byte> w, int* act)
    {
        Vector256<byte> u = Avx2.Xor(w, Vector256.Create((byte)0x80));
        Vector256<int> dots = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, u, Avx2.BroadcastScalarToVector256(act).AsSByte());
        sbyte* p = (sbyte*)act;
        int sum4 = p[0] + p[1] + p[2] + p[3];
        return Avx2.Subtract(dots, Vector256.Create(sum4 << 7));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmTiles(int n, float* dst, int ldc, BlockQ8_0x8* weights, BlockQ8_0x4* activations, int rows, int cols, bool vnni)
    {
        int nb = n / Qk.Q8_0Block;
        int tileGroups = Math.Max(1, GemmQ4K.ColTileBytes / (nb * Qk.Q8_0x8Size));
        int groups = cols / 8;
        for (int g0 = 0; g0 < groups; g0 += tileGroups)
        {
            int g1 = Math.Min(groups, g0 + tileGroups);
            if (vnni)
                GemmTileVnni(n, dst + g0 * 8, ldc, weights + g0 * nb, activations, rows, (g1 - g0) * 8);
            else
                GemmTileAvx2(n, dst + g0 * 8, ldc, weights + g0 * nb, activations, rows, (g1 - g0) * 8);
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmTileAvx2(int n, float* dst, int ldc, BlockQ8_0x8* weights, BlockQ8_0x4* activations, int rows, int cols)
    {
        if ((rows & 7) == 0)
        {
            GemmTileAvx28(n, dst, ldc, weights, activations, rows, cols);
            return;
        }

        GemmTileAvx24(n, dst, ldc, weights, activations, rows, cols);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmTileAvx28(int n, float* dst, int ldc, BlockQ8_0x8* weights, BlockQ8_0x4* activations, int rows, int cols)
    {
        int nb = n / Qk.Q8_0Block;
        Vector256<short> ones = Vector256<short>.One;
        for (int y = 0; y < rows / 4; y += 2)
        {
            BlockQ8_0x4* a0 = activations + y * nb;
            BlockQ8_0x4* a1 = a0 + nb;
            for (int x = 0; x < cols / 8; x++)
            {
                BlockQ8_0x8* bPtr = weights + x * nb;
                Vector256<float> acc0 = Vector256<float>.Zero;
                Vector256<float> acc1 = Vector256<float>.Zero;
                Vector256<float> acc2 = Vector256<float>.Zero;
                Vector256<float> acc3 = Vector256<float>.Zero;
                Vector256<float> acc4 = Vector256<float>.Zero;
                Vector256<float> acc5 = Vector256<float>.Zero;
                Vector256<float> acc6 = Vector256<float>.Zero;
                Vector256<float> acc7 = Vector256<float>.Zero;
                for (int b = 0; b < nb; b++)
                {
                    BlockQ8_0x4* p0 = a0 + b;
                    BlockQ8_0x4* p1 = a1 + b;
                    Vector256<int> j0 = Vector256<int>.Zero;
                    Vector256<int> j1 = Vector256<int>.Zero;
                    Vector256<int> j2 = Vector256<int>.Zero;
                    Vector256<int> j3 = Vector256<int>.Zero;
                    Vector256<int> j4 = Vector256<int>.Zero;
                    Vector256<int> j5 = Vector256<int>.Zero;
                    Vector256<int> j6 = Vector256<int>.Zero;
                    Vector256<int> j7 = Vector256<int>.Zero;
                    byte* q = bPtr[b].Qs;
                    sbyte* r0 = p0->Qs;
                    sbyte* r1 = p1->Qs;
                    for (int step = 0; step < 8; step++)
                    {
                        Vector256<sbyte> ws = Avx.LoadVector256(q + step * 32).AsSByte();
                        Vector256<byte> aw = Avx2.Sign(ws, ws).AsByte();
                        j0 = Avx2.Add(j0, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(aw, Avx2.Sign(Avx2.BroadcastScalarToVector256((int*)(r0 + step * 4)).AsSByte(), ws)), ones));
                        j1 = Avx2.Add(j1, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(aw, Avx2.Sign(Avx2.BroadcastScalarToVector256((int*)(r0 + 32 + step * 4)).AsSByte(), ws)), ones));
                        j2 = Avx2.Add(j2, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(aw, Avx2.Sign(Avx2.BroadcastScalarToVector256((int*)(r0 + 64 + step * 4)).AsSByte(), ws)), ones));
                        j3 = Avx2.Add(j3, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(aw, Avx2.Sign(Avx2.BroadcastScalarToVector256((int*)(r0 + 96 + step * 4)).AsSByte(), ws)), ones));
                        j4 = Avx2.Add(j4, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(aw, Avx2.Sign(Avx2.BroadcastScalarToVector256((int*)(r1 + step * 4)).AsSByte(), ws)), ones));
                        j5 = Avx2.Add(j5, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(aw, Avx2.Sign(Avx2.BroadcastScalarToVector256((int*)(r1 + 32 + step * 4)).AsSByte(), ws)), ones));
                        j6 = Avx2.Add(j6, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(aw, Avx2.Sign(Avx2.BroadcastScalarToVector256((int*)(r1 + 64 + step * 4)).AsSByte(), ws)), ones));
                        j7 = Avx2.Add(j7, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(aw, Avx2.Sign(Avx2.BroadcastScalarToVector256((int*)(r1 + 96 + step * 4)).AsSByte(), ws)), ones));
                    }

                    Vector256<float> col = Avx.LoadVector256(bPtr[b].D);
                    acc0 = ScaleAdd(acc0, j0, col, p0->D[0]);
                    acc1 = ScaleAdd(acc1, j1, col, p0->D[1]);
                    acc2 = ScaleAdd(acc2, j2, col, p0->D[2]);
                    acc3 = ScaleAdd(acc3, j3, col, p0->D[3]);
                    acc4 = ScaleAdd(acc4, j4, col, p1->D[0]);
                    acc5 = ScaleAdd(acc5, j5, col, p1->D[1]);
                    acc6 = ScaleAdd(acc6, j6, col, p1->D[2]);
                    acc7 = ScaleAdd(acc7, j7, col, p1->D[3]);
                }

                int row = y * 4;
                Avx.Store(dst + (row + 0) * ldc + x * 8, acc0);
                Avx.Store(dst + (row + 1) * ldc + x * 8, acc1);
                Avx.Store(dst + (row + 2) * ldc + x * 8, acc2);
                Avx.Store(dst + (row + 3) * ldc + x * 8, acc3);
                Avx.Store(dst + (row + 4) * ldc + x * 8, acc4);
                Avx.Store(dst + (row + 5) * ldc + x * 8, acc5);
                Avx.Store(dst + (row + 6) * ldc + x * 8, acc6);
                Avx.Store(dst + (row + 7) * ldc + x * 8, acc7);
            }
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmTileAvx24(int n, float* dst, int ldc, BlockQ8_0x8* weights, BlockQ8_0x4* activations, int rows, int cols)
    {
        int nb = n / Qk.Q8_0Block;
        for (int y = 0; y < rows / 4; y++)
        {
            BlockQ8_0x4* aPtr = activations + y * nb;
            for (int x = 0; x < cols / 8; x++)
            {
                BlockQ8_0x8* bPtr = weights + x * nb;
                Vector256<float> acc0 = Vector256<float>.Zero;
                Vector256<float> acc1 = Vector256<float>.Zero;
                Vector256<float> acc2 = Vector256<float>.Zero;
                Vector256<float> acc3 = Vector256<float>.Zero;
                for (int b = 0; b < nb; b++)
                {
                    BlockQ8_0x4* ab = aPtr + b;
                    BlockQ8_0x8* wb = bPtr + b;
                    Vector256<int> j0 = Vector256<int>.Zero;
                    Vector256<int> j1 = Vector256<int>.Zero;
                    Vector256<int> j2 = Vector256<int>.Zero;
                    Vector256<int> j3 = Vector256<int>.Zero;
                    byte* q = wb->Qs;
                    sbyte* qs = ab->Qs;
                    for (int step = 0; step < 8; step++)
                    {
                        Vector256<byte> w = Avx.LoadVector256(q + step * 32);
                        j0 = Avx2.Add(j0, Dot4Avx2(w, (int*)(qs + step * 4)));
                        j1 = Avx2.Add(j1, Dot4Avx2(w, (int*)(qs + 32 + step * 4)));
                        j2 = Avx2.Add(j2, Dot4Avx2(w, (int*)(qs + 64 + step * 4)));
                        j3 = Avx2.Add(j3, Dot4Avx2(w, (int*)(qs + 96 + step * 4)));
                    }

                    Vector256<float> col = Avx.LoadVector256(wb->D);
                    acc0 = ScaleAdd(acc0, j0, col, ab->D[0]);
                    acc1 = ScaleAdd(acc1, j1, col, ab->D[1]);
                    acc2 = ScaleAdd(acc2, j2, col, ab->D[2]);
                    acc3 = ScaleAdd(acc3, j3, col, ab->D[3]);
                }

                Avx.Store(dst + (y * 4 + 0) * ldc + x * 8, acc0);
                Avx.Store(dst + (y * 4 + 1) * ldc + x * 8, acc1);
                Avx.Store(dst + (y * 4 + 2) * ldc + x * 8, acc2);
                Avx.Store(dst + (y * 4 + 3) * ldc + x * 8, acc3);
            }
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmTileVnni(int n, float* dst, int ldc, BlockQ8_0x8* weights, BlockQ8_0x4* activations, int rows, int cols)
    {
        if ((rows & 7) == 0)
        {
            GemmTileVnni8(n, dst, ldc, weights, activations, rows, cols);
            return;
        }

        GemmTileVnni4(n, dst, ldc, weights, activations, rows, cols);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmTileVnni8(int n, float* dst, int ldc, BlockQ8_0x8* weights, BlockQ8_0x4* activations, int rows, int cols)
    {
        int nb = n / Qk.Q8_0Block;
        Vector256<byte> bias = Vector256.Create((byte)0x80);
        for (int y = 0; y < rows / 4; y += 2)
        {
            BlockQ8_0x4* a0 = activations + y * nb;
            BlockQ8_0x4* a1 = a0 + nb;
            for (int x = 0; x < cols / 8; x++)
            {
                BlockQ8_0x8* bPtr = weights + x * nb;
                Vector256<float> acc0 = Vector256<float>.Zero;
                Vector256<float> acc1 = Vector256<float>.Zero;
                Vector256<float> acc2 = Vector256<float>.Zero;
                Vector256<float> acc3 = Vector256<float>.Zero;
                Vector256<float> acc4 = Vector256<float>.Zero;
                Vector256<float> acc5 = Vector256<float>.Zero;
                Vector256<float> acc6 = Vector256<float>.Zero;
                Vector256<float> acc7 = Vector256<float>.Zero;
                for (int b = 0; b < nb; b++)
                {
                    BlockQ8_0x4* p0 = a0 + b;
                    BlockQ8_0x4* p1 = a1 + b;
                    Vector256<int> j0 = Vector256<int>.Zero;
                    Vector256<int> j1 = Vector256<int>.Zero;
                    Vector256<int> j2 = Vector256<int>.Zero;
                    Vector256<int> j3 = Vector256<int>.Zero;
                    Vector256<int> j4 = Vector256<int>.Zero;
                    Vector256<int> j5 = Vector256<int>.Zero;
                    Vector256<int> j6 = Vector256<int>.Zero;
                    Vector256<int> j7 = Vector256<int>.Zero;
                    byte* q = bPtr[b].Qs;
                    sbyte* r0 = p0->Qs;
                    sbyte* r1 = p1->Qs;
                    for (int step = 0; step < 8; step++)
                    {
                        Vector256<byte> u = Avx2.Xor(Avx.LoadVector256(q + step * 32), bias);
                        j0 = AvxVnni.MultiplyWideningAndAdd(j0, u, Avx2.BroadcastScalarToVector256((int*)(r0 + step * 4)).AsSByte());
                        j1 = AvxVnni.MultiplyWideningAndAdd(j1, u, Avx2.BroadcastScalarToVector256((int*)(r0 + 32 + step * 4)).AsSByte());
                        j2 = AvxVnni.MultiplyWideningAndAdd(j2, u, Avx2.BroadcastScalarToVector256((int*)(r0 + 64 + step * 4)).AsSByte());
                        j3 = AvxVnni.MultiplyWideningAndAdd(j3, u, Avx2.BroadcastScalarToVector256((int*)(r0 + 96 + step * 4)).AsSByte());
                        j4 = AvxVnni.MultiplyWideningAndAdd(j4, u, Avx2.BroadcastScalarToVector256((int*)(r1 + step * 4)).AsSByte());
                        j5 = AvxVnni.MultiplyWideningAndAdd(j5, u, Avx2.BroadcastScalarToVector256((int*)(r1 + 32 + step * 4)).AsSByte());
                        j6 = AvxVnni.MultiplyWideningAndAdd(j6, u, Avx2.BroadcastScalarToVector256((int*)(r1 + 64 + step * 4)).AsSByte());
                        j7 = AvxVnni.MultiplyWideningAndAdd(j7, u, Avx2.BroadcastScalarToVector256((int*)(r1 + 96 + step * 4)).AsSByte());
                    }

                    Vector256<float> col = Avx.LoadVector256(bPtr[b].D);
                    acc0 = ScaleAdd(acc0, Avx2.Subtract(j0, Vector256.Create(p0->Bias[0])), col, p0->D[0]);
                    acc1 = ScaleAdd(acc1, Avx2.Subtract(j1, Vector256.Create(p0->Bias[1])), col, p0->D[1]);
                    acc2 = ScaleAdd(acc2, Avx2.Subtract(j2, Vector256.Create(p0->Bias[2])), col, p0->D[2]);
                    acc3 = ScaleAdd(acc3, Avx2.Subtract(j3, Vector256.Create(p0->Bias[3])), col, p0->D[3]);
                    acc4 = ScaleAdd(acc4, Avx2.Subtract(j4, Vector256.Create(p1->Bias[0])), col, p1->D[0]);
                    acc5 = ScaleAdd(acc5, Avx2.Subtract(j5, Vector256.Create(p1->Bias[1])), col, p1->D[1]);
                    acc6 = ScaleAdd(acc6, Avx2.Subtract(j6, Vector256.Create(p1->Bias[2])), col, p1->D[2]);
                    acc7 = ScaleAdd(acc7, Avx2.Subtract(j7, Vector256.Create(p1->Bias[3])), col, p1->D[3]);
                }

                int row = y * 4;
                Avx.Store(dst + (row + 0) * ldc + x * 8, acc0);
                Avx.Store(dst + (row + 1) * ldc + x * 8, acc1);
                Avx.Store(dst + (row + 2) * ldc + x * 8, acc2);
                Avx.Store(dst + (row + 3) * ldc + x * 8, acc3);
                Avx.Store(dst + (row + 4) * ldc + x * 8, acc4);
                Avx.Store(dst + (row + 5) * ldc + x * 8, acc5);
                Avx.Store(dst + (row + 6) * ldc + x * 8, acc6);
                Avx.Store(dst + (row + 7) * ldc + x * 8, acc7);
            }
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmTileVnni4(int n, float* dst, int ldc, BlockQ8_0x8* weights, BlockQ8_0x4* activations, int rows, int cols)
    {
        int nb = n / Qk.Q8_0Block;
        Vector256<byte> bias = Vector256.Create((byte)0x80);
        for (int y = 0; y < rows / 4; y++)
        {
            BlockQ8_0x4* aPtr = activations + y * nb;
            for (int x = 0; x < cols / 8; x++)
            {
                BlockQ8_0x8* bPtr = weights + x * nb;
                Vector256<float> acc0 = Vector256<float>.Zero;
                Vector256<float> acc1 = Vector256<float>.Zero;
                Vector256<float> acc2 = Vector256<float>.Zero;
                Vector256<float> acc3 = Vector256<float>.Zero;
                for (int b = 0; b < nb; b++)
                {
                    BlockQ8_0x4* ab = aPtr + b;
                    BlockQ8_0x8* wb = bPtr + b;
                    Vector256<int> j0 = Vector256<int>.Zero;
                    Vector256<int> j1 = Vector256<int>.Zero;
                    Vector256<int> j2 = Vector256<int>.Zero;
                    Vector256<int> j3 = Vector256<int>.Zero;
                    byte* q = wb->Qs;
                    sbyte* qs = ab->Qs;
                    for (int step = 0; step < 8; step++)
                    {
                        Vector256<byte> u = Avx2.Xor(Avx.LoadVector256(q + step * 32), bias);
                        j0 = AvxVnni.MultiplyWideningAndAdd(j0, u, Avx2.BroadcastScalarToVector256((int*)(qs + step * 4)).AsSByte());
                        j1 = AvxVnni.MultiplyWideningAndAdd(j1, u, Avx2.BroadcastScalarToVector256((int*)(qs + 32 + step * 4)).AsSByte());
                        j2 = AvxVnni.MultiplyWideningAndAdd(j2, u, Avx2.BroadcastScalarToVector256((int*)(qs + 64 + step * 4)).AsSByte());
                        j3 = AvxVnni.MultiplyWideningAndAdd(j3, u, Avx2.BroadcastScalarToVector256((int*)(qs + 96 + step * 4)).AsSByte());
                    }

                    j0 = Avx2.Subtract(j0, Vector256.Create(ab->Bias[0]));
                    j1 = Avx2.Subtract(j1, Vector256.Create(ab->Bias[1]));
                    j2 = Avx2.Subtract(j2, Vector256.Create(ab->Bias[2]));
                    j3 = Avx2.Subtract(j3, Vector256.Create(ab->Bias[3]));
                    Vector256<float> col = Avx.LoadVector256(wb->D);
                    acc0 = ScaleAdd(acc0, j0, col, ab->D[0]);
                    acc1 = ScaleAdd(acc1, j1, col, ab->D[1]);
                    acc2 = ScaleAdd(acc2, j2, col, ab->D[2]);
                    acc3 = ScaleAdd(acc3, j3, col, ab->D[3]);
                }

                Avx.Store(dst + (y * 4 + 0) * ldc + x * 8, acc0);
                Avx.Store(dst + (y * 4 + 1) * ldc + x * 8, acc1);
                Avx.Store(dst + (y * 4 + 2) * ldc + x * 8, acc2);
                Avx.Store(dst + (y * 4 + 3) * ldc + x * 8, acc3);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ScaleAdd(Vector256<float> acc, Vector256<int> dots, Vector256<float> colScale, float rowScale)
    {
        Vector256<float> scale = Avx.Multiply(colScale, Vector256.Create(rowScale));
        Vector256<float> v = Avx.ConvertToVector256Single(dots);
        return Fma.IsSupported ? Fma.MultiplyAdd(v, scale, acc) : Avx.Add(acc, Avx.Multiply(v, scale));
    }
}
