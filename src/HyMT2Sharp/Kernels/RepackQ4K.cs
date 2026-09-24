namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class RepackQ4K
{
    public static void MakeBlockX8(BlockQ4K* input, BlockQ4Kx8* output, int interleaveBytes = 8)
    {
        for (int i = 0; i < 8; i++)
        {
            output->D[i] = input[i].D;
            output->Dmin[i] = input[i].Dmin;
        }

        int end = Qk.SuperBlock * 4 / interleaveBytes;
        for (int i = 0; i < end; i++)
        {
            int srcId = i % 8;
            int srcOffset = (i / 8) * interleaveBytes;
            int dstOffset = i * interleaveBytes;
            ulong elems = 0;
            Buffer.MemoryCopy(input[srcId].Qs + srcOffset, &elems, interleaveBytes, interleaveBytes);
            Buffer.MemoryCopy(&elems, output->Qs + dstOffset, interleaveBytes, interleaveBytes);
        }

        byte* s = stackalloc byte[8];
        byte* m = stackalloc byte[8];
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                s[j] = (byte)(input[j].Scales[i] & 63);
                m[j] = (byte)(input[j].Scales[i + 4] & 63);
            }

            output->Scales[i * 12] = (byte)((s[0] & 63) + ((s[4] & 48) << 2));
            output->Scales[i * 12 + 1] = (byte)((s[1] & 63) + ((s[5] & 48) << 2));
            output->Scales[i * 12 + 2] = (byte)((s[2] & 63) + ((s[6] & 48) << 2));
            output->Scales[i * 12 + 3] = (byte)((s[3] & 63) + ((s[7] & 48) << 2));
            output->Scales[i * 12 + 4] = (byte)((m[0] & 63) + ((m[4] & 48) << 2));
            output->Scales[i * 12 + 5] = (byte)((m[1] & 63) + ((m[5] & 48) << 2));
            output->Scales[i * 12 + 6] = (byte)((m[2] & 63) + ((m[6] & 48) << 2));
            output->Scales[i * 12 + 7] = (byte)((m[3] & 63) + ((m[7] & 48) << 2));
            output->Scales[i * 12 + 8] = (byte)((s[4] & 15) + ((m[4] & 15) << 4));
            output->Scales[i * 12 + 9] = (byte)((s[5] & 15) + ((m[5] & 15) << 4));
            output->Scales[i * 12 + 10] = (byte)((s[6] & 15) + ((m[6] & 15) << 4));
            output->Scales[i * 12 + 11] = (byte)((s[7] & 15) + ((m[7] & 15) << 4));
        }

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                s[j] = (byte)(((input[j].Scales[i] & 192) >> 2) | (input[j].Scales[i + 8] & 15));
                m[j] = (byte)(((input[j].Scales[i + 4] & 192) >> 2) | ((input[j].Scales[i + 8] & 240) >> 4));
            }

            output->Scales[i * 12 + 48] = (byte)((s[0] & 63) + ((s[4] & 48) << 2));
            output->Scales[i * 12 + 49] = (byte)((s[1] & 63) + ((s[5] & 48) << 2));
            output->Scales[i * 12 + 50] = (byte)((s[2] & 63) + ((s[6] & 48) << 2));
            output->Scales[i * 12 + 51] = (byte)((s[3] & 63) + ((s[7] & 48) << 2));
            output->Scales[i * 12 + 52] = (byte)((m[0] & 63) + ((m[4] & 48) << 2));
            output->Scales[i * 12 + 53] = (byte)((m[1] & 63) + ((m[5] & 48) << 2));
            output->Scales[i * 12 + 54] = (byte)((m[2] & 63) + ((m[6] & 48) << 2));
            output->Scales[i * 12 + 55] = (byte)((m[3] & 63) + ((m[7] & 48) << 2));
            output->Scales[i * 12 + 56] = (byte)((s[4] & 15) + ((m[4] & 15) << 4));
            output->Scales[i * 12 + 57] = (byte)((s[5] & 15) + ((m[5] & 15) << 4));
            output->Scales[i * 12 + 58] = (byte)((s[6] & 15) + ((m[6] & 15) << 4));
            output->Scales[i * 12 + 59] = (byte)((s[7] & 15) + ((m[7] & 15) << 4));
        }
    }

    public static void Rows(BlockQ4K* src, BlockQ4Kx8* dst, int nIn, int nOut)
    {
        int nb = nIn / Qk.SuperBlock;
        int groups = nOut / 8;
        BlockQ4K* tmp = stackalloc BlockQ4K[8];
        for (int g = 0; g < groups; g++)
        {
            for (int b = 0; b < nb; b++)
            {
                for (int r = 0; r < 8; r++)
                    tmp[r] = src[(g * 8 + r) * nb + b];
                MakeBlockX8(tmp, &dst[g * nb + b]);
            }
        }
    }

    /// <summary>
    /// Builds the <see cref="BlockQ4Kx8Meta"/> sidecar (f32 d/dmin plus pre-decoded int16
    /// scales/mins in kernel order) for every packed block of a row-major Q4_K weight.
    /// </summary>
    public static void BuildMeta(BlockQ4K* src, BlockQ4Kx8Meta* dst, int nIn, int nOut)
    {
        if (Simd.UseDp && !Simd.UseAvx2)
        {
            BuildMetaNeon(src, dst, nIn, nOut);
            return;
        }

        int nb = nIn / Qk.SuperBlock;
        int groups = nOut / 8;
        // vphaddw(cols0123, cols4567) lane order.
        ReadOnlySpan<byte> order = [0, 0, 1, 1, 4, 4, 5, 5, 2, 2, 3, 3, 6, 6, 7, 7];
        byte* sc = stackalloc byte[8];
        byte* mn = stackalloc byte[8];
        for (int g = 0; g < groups; g++)
        {
            for (int b = 0; b < nb; b++)
            {
                BlockQ4Kx8Meta* meta = dst + g * nb + b;
                for (int j = 0; j < 8; j++)
                {
                    BlockQ4K* block = src + (g * 8 + j) * nb + b;
                    meta->D[j] = HalfBits.ToSingle(block->D);
                    meta->Dmin[j] = HalfBits.ToSingle(block->Dmin);
                }

                for (int pair = 0; pair < 4; pair++)
                {
                    short* lo = meta->Scales + pair * 48;
                    short* hi = lo + 16;
                    short* mins = lo + 32;
                    for (int half = 0; half < 2; half++)
                    {
                        int sub = pair * 2 + half;
                        for (int j = 0; j < 8; j++)
                            ScaleMin(sub, (src + (g * 8 + j) * nb + b)->Scales, out sc[j], out mn[j]);
                        short* target = half == 0 ? lo : hi;
                        for (int i = 0; i < 16; i++)
                            target[i] = sc[order[i]];
                        for (int j = 0; j < 8; j++)
                            mins[j * 2 + half] = mn[j];
                    }
                }
            }
        }
    }

    /// <summary>
    /// Natural-column variant of <see cref="BuildMeta"/> for the SDOT kernel:
    /// <c>Scales[s * 8 + c]</c> holds sub-block s scales and <c>Scales[64 + s * 8 + c]</c>
    /// its mins, so the kernel reads each with a single 16-byte load.
    /// </summary>
    private static void BuildMetaNeon(BlockQ4K* src, BlockQ4Kx8Meta* dst, int nIn, int nOut)
    {
        int nb = nIn / Qk.SuperBlock;
        int groups = nOut / 8;
        for (int g = 0; g < groups; g++)
        {
            for (int b = 0; b < nb; b++)
            {
                BlockQ4Kx8Meta* meta = dst + g * nb + b;
                for (int j = 0; j < 8; j++)
                {
                    BlockQ4K* block = src + (g * 8 + j) * nb + b;
                    meta->D[j] = HalfBits.ToSingle(block->D);
                    meta->Dmin[j] = HalfBits.ToSingle(block->Dmin);
                }

                for (int s = 0; s < 8; s++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        ScaleMin(s, (src + (g * 8 + j) * nb + b)->Scales, out byte d, out byte m);
                        meta->Scales[s * 8 + j] = d;
                        meta->Scales[64 + s * 8 + j] = m;
                    }
                }
            }
        }
    }

    /// <summary>ggml <c>get_scale_min_k4</c>.</summary>
    private static void ScaleMin(int j, byte* q, out byte d, out byte m)
    {
        if (j < 4)
        {
            d = (byte)(q[j] & 63);
            m = (byte)(q[j + 4] & 63);
        }
        else
        {
            d = (byte)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            m = (byte)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }
}
