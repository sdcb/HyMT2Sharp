namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class RepackQ8_0
{
    public static void Rows(BlockQ8_0* src, BlockQ8_0x8* dst, int nIn, int nOut)
    {
        int nb = nIn / Qk.Q8_0Block;
        int groups = nOut / 8;
        for (int g = 0; g < groups; g++)
        {
            for (int b = 0; b < nb; b++)
            {
                BlockQ8_0x8* panel = &dst[g * nb + b];
                for (int c = 0; c < 8; c++)
                {
                    BlockQ8_0* row = src + (g * 8 + c) * nb + b;
                    panel->D[c] = HalfBits.ToSingle(row->D);
                    for (int k = 0; k < Qk.Q8_0Block; k++)
                        panel->Qs[(k >> 2) * 32 + c * 4 + (k & 3)] = (byte)row->Qs[k];
                }
            }
        }
    }
}
