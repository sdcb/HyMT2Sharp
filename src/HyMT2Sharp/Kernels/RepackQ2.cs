namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class RepackQ2
{
    public static void Rows(BlockQ2_0C* src, BlockQ2x8* dst, int nIn, int nOut)
    {
        if (nIn % Q2_0C.BlockLength != 0)
            throw new ArgumentException("Q2 rows must contain a multiple of 512 values.", nameof(nIn));
        int nb = nIn / Q2_0C.BlockLength;
        for (int g = 0; g < nOut / 8; g++)
            for (int b = 0; b < nb; b++)
            {
                BlockQ2x8* panel = dst + g * nb + b;
                for (int c = 0; c < 8; c++)
                {
                    BlockQ2_0C* row = src + (g * 8 + c) * nb + b;
                    panel->D[c] = HalfBits.ToSingle(row->D);
                    for (int k = 0; k < 512; k += 32)
                        for (int i = 0; i < 8; i++)
                        {
                            int code = 0;
                            for (int p = 0; p < 4; p++)
                            {
                                int at = k + p * 8 + i;
                                code |= ((row->Qs[at / 4] >> ((at & 3) * 2)) & 3) << (p * 2);
                            }
                            panel->Qs[(k / 32) * 64 + c * 8 + i] = (byte)code;
                        }
                }
            }
    }
}
