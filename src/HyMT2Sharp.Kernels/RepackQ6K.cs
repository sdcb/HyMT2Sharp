namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class RepackQ6K
{
    /// <summary>Expand one Q6_K block to 256 unsigned 6-bit values (before the −32 offset).</summary>
    public static void Expand(BlockQ6K* block, byte* values)
    {
        byte* ql = block->Ql;
        byte* qh = block->Qh;
        for (int n = 0; n < Qk.SuperBlock; n += 128)
        {
            for (int l = 0; l < 32; l++)
            {
                values[n + l] = (byte)((ql[l] & 0xF) | ((qh[l] & 3) << 4));
                values[n + l + 32] = (byte)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4));
                values[n + l + 64] = (byte)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4));
                values[n + l + 96] = (byte)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4));
            }

            ql += 64;
            qh += 32;
        }
    }

    public static void MakeBlockX8(BlockQ6K* input, BlockQ6Kx8* output)
    {
        byte* values = stackalloc byte[Qk.SuperBlock];
        for (int c = 0; c < 8; c++)
        {
            output->D[c] = HalfBits.ToSingle(input[c].D);
            Expand(input + c, values);
            for (int k = 0; k < Qk.SuperBlock; k++)
                output->Qs[(k >> 2) * 32 + c * 4 + (k & 3)] = values[k];
            for (int i = 0; i < 16; i++)
            {
                short sc = input[c].Scales[i];
                output->Scales[i * 16 + c * 2] = sc;
                output->Scales[i * 16 + c * 2 + 1] = sc;
                output->ScalePairs[(i >> 1) * 16 + c * 2 + (i & 1)] = sc;
            }
        }
    }

    /// <summary>
    /// NEON variant: weights are stored pre-offset (code − 32, signed) so the SDOT
    /// kernel needs no bsum correction, and scales use natural column order
    /// <c>Scales[i * 8 + c]</c> instead of the vphaddw pairing.
    /// </summary>
    public static void MakeBlockX8Neon(BlockQ6K* input, BlockQ6Kx8* output)
    {
        byte* values = stackalloc byte[Qk.SuperBlock];
        for (int c = 0; c < 8; c++)
        {
            output->D[c] = HalfBits.ToSingle(input[c].D);
            Expand(input + c, values);
            for (int k = 0; k < Qk.SuperBlock; k++)
                output->Qs[(k >> 2) * 32 + c * 4 + (k & 3)] = (byte)(values[k] - 32);
            for (int i = 0; i < 16; i++)
                output->Scales[i * 8 + c] = input[c].Scales[i];
        }
    }

    /// <summary>Canonical (AVX2-order) panel repack; also read by <see cref="GemmQ6K.GemmScalar"/>.</summary>
    public static void Rows(BlockQ6K* src, BlockQ6Kx8* dst, int nIn, int nOut) => RowsImpl(src, dst, nIn, nOut, neon: false);

    /// <summary>NEON panel repack (signed weights, natural scales) for <see cref="GemmQ6K.GemmNeon"/>.</summary>
    public static void RowsNeon(BlockQ6K* src, BlockQ6Kx8* dst, int nIn, int nOut) => RowsImpl(src, dst, nIn, nOut, neon: true);

    private static void RowsImpl(BlockQ6K* src, BlockQ6Kx8* dst, int nIn, int nOut, bool neon)
    {
        int nb = nIn / Qk.SuperBlock;
        int groups = nOut / 8;
        BlockQ6K* tmp = stackalloc BlockQ6K[8];
        for (int g = 0; g < groups; g++)
        {
            for (int b = 0; b < nb; b++)
            {
                for (int r = 0; r < 8; r++)
                    tmp[r] = src[(g * 8 + r) * nb + b];
                if (neon)
                    MakeBlockX8Neon(tmp, &dst[g * nb + b]);
                else
                    MakeBlockX8(tmp, &dst[g * nb + b]);
            }
        }
    }
}
