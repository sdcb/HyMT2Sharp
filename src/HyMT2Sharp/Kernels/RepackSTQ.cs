using System.Runtime.CompilerServices;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Repack legacy STQ rows into an eight-column panel.  The source stores a
/// codebook slot for every stride-16 group; the panel stores the decoded 2-bit
/// values (0, 1, 2) in the same 32-value tile order as <see cref="Q2Panel"/>.
/// This conversion is paid once while loading the model and keeps the hot
/// GEMV/GEMM loops free of nibble/sign unpacking.
/// </summary>
public static unsafe class RepackSTQ
{
    public static void Rows(BlockSTQ1_0* src, BlockSTQ1_0x8* dst, int nIn, int nOut)
    {
        if (nIn <= 0 || nIn % STQ1_0.BlockLength != 0)
            throw new ArgumentException("STQ rows require a positive multiple of 256 inputs.", nameof(nIn));
        if (nOut < 8 || (nOut & 7) != 0)
            throw new ArgumentException("STQ panels require at least eight rows and an output count divisible by eight.", nameof(nOut));

        int nb = nIn / STQ1_0.BlockLength;
        int groups = nOut / 8;
        for (int g = 0; g < groups; g++)
        {
            for (int b = 0; b < nb; b++)
            {
                BlockSTQ1_0x8* panel = dst + (long)g * nb + b;
                for (int i = 0; i < 512; i++)
                    panel->Qs[i] = 0;
                for (int c = 0; c < 8; c++)
                {
                    BlockSTQ1_0* row = src + (long)(g * 8 + c) * nb + b;
                    panel->D[c] = HalfBits.ToSingle(row->D);
                    for (int group = 0; group < STQ1_0.BlockLength / 4; group++)
                    {
                        byte qpack = STQ1_0.QPack(row, group);
                        int chunk = group >> 4;
                        int gloc = group & 15;
                        for (int lane = 0; lane < 4; lane++)
                        {
                            int pos = chunk * 64 + gloc + lane * 16;
                            int tile = pos >> 5;
                            int plane = (pos >> 3) & 3;
                            int index = tile * 64 + c * 8 + (pos & 7);
                            panel->Qs[index] |= (byte)(((qpack >> (lane * 2)) & 3) << (plane * 2));
                        }
                    }
                }
            }
        }
    }
}
