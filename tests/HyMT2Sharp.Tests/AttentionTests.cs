using Sdcb.HyMT2Sharp.Model;

namespace Sdcb.HyMT2Sharp.Tests;

public sealed class AttentionTests
{
    [Fact]
    public unsafe void ScoresAndCombine_MatchScalar()
    {
        const int heads = 4;
        const int kvHeads = 2;
        const int dim = 32;
        const int qLen = 3;
        const int kvLen = 5;
        const int qDim = heads * dim;
        const int kvStride = kvHeads * dim;
        float scale = 1f / MathF.Sqrt(dim);

        float[] q = Random(qLen * qDim, 1);
        ushort[] cacheK = Random(kvLen * kvStride, 2).Select(ToBf16).ToArray();
        ushort[] cacheV = Random(kvLen * kvStride, 3).Select(ToBf16).ToArray();
        float[] scores = new float[heads * qLen * kvLen];
        float[] scoresRef = new float[heads * qLen * kvLen];
        float[] ao = new float[qLen * qDim];
        float[] aoRef = new float[qLen * qDim];

        fixed (float* qp = q, sc = scores, scr = scoresRef, aop = ao, aor = aoRef)
        fixed (ushort* kp = cacheK, vp = cacheV)
        {
            int group = heads / kvHeads;
            for (int h = 0; h < heads; h++)
            {
                int kvh = h / group;
                for (int qt = 0; qt < qLen; qt++)
                {
                    float* qh = qp + qt * qDim + h * dim;
                    float* row = scr + (h * qLen + qt) * kvLen;
                    for (int kt = 0; kt < kvLen; kt++)
                    {
                        ushort* kh = kp + kt * kvStride + kvh * dim;
                        float dot = 0;
                        for (int i = 0; i < dim; i++)
                            dot += qh[i] * B2F(kh[i]);
                        row[kt] = dot * scale;
                    }
                }
            }

            Ops.AttentionScores(qp, kp, sc, heads, kvHeads, dim, qLen, kvLen, qDim, kvStride, scale, kvLen);
            AssertClose(scr, sc, scores.Length, 5e-5f);

            for (int h = 0; h < heads; h++)
            {
                int kvh = h / group;
                for (int qt = 0; qt < qLen; qt++)
                {
                    float* row = scr + (h * qLen + qt) * kvLen;
                    float* outH = aor + qt * qDim + h * dim;
                    for (int kt = 0; kt < kvLen; kt++)
                    {
                        float w = row[kt];
                        ushort* vh = vp + kt * kvStride + kvh * dim;
                        for (int i = 0; i < dim; i++)
                            outH[i] += w * B2F(vh[i]);
                    }
                }
            }

            Ops.AttentionCombine(vp, sc, aop, heads, kvHeads, dim, qLen, kvLen, qDim, kvStride, kvLen);
            AssertClose(aor, aop, ao.Length, 5e-4f);
        }
    }

    [Fact]
    public unsafe void SoftmaxCausal_MatchesScalarExp()
    {
        const int heads = 2;
        const int qLen = 5;
        const int kvLen = 8;
        float[] scores = Random(heads * qLen * kvLen, 8);
        float[] expected = (float[])scores.Clone();
        fixed (float* sc = scores, ex = expected)
        {
            for (int h = 0; h < heads; h++)
            {
                for (int q = 0; q < qLen; q++)
                {
                    float* row = ex + (h * qLen + q) * kvLen;
                    int allowed = q + 1;
                    float max = float.NegativeInfinity;
                    for (int k = 0; k < kvLen; k++)
                    {
                        if (k >= allowed)
                            row[k] = float.NegativeInfinity;
                        if (row[k] > max)
                            max = row[k];
                    }

                    float sum = 0;
                    for (int k = 0; k < kvLen; k++)
                    {
                        float e = float.IsNegativeInfinity(row[k]) ? 0 : MathF.Exp(row[k] - max);
                        row[k] = e;
                        sum += e;
                    }

                    float inv = sum > 0 ? 1f / sum : 0;
                    for (int k = 0; k < kvLen; k++)
                        row[k] *= inv;
                }
            }

            Ops.SoftmaxCausal(sc, heads, qLen, kvLen, 0);
            AssertClose(ex, sc, scores.Length, 2e-3f);
        }
    }

    [Fact]
    public unsafe void DotF32_MatchesScalar()
    {
        const int n = 128;
        float[] a = Random(n, 4);
        float[] b = Random(n, 5);
        float expected = 0;
        for (int i = 0; i < n; i++)
            expected += a[i] * b[i];
        fixed (float* ap = a, bp = b)
        {
            float got = Ops.DotF32(ap, bp, n);
            Assert.True(MathF.Abs(got - expected) < 1e-4f * n, $"{got} vs {expected}");
        }
    }

    private static ushort ToBf16(float x)
    {
        uint bits = BitConverter.SingleToUInt32Bits(x);
        return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
    }

    private static float B2F(ushort h) => BitConverter.UInt32BitsToSingle((uint)h << 16);

    private static float[] Random(int n, int seed)
    {
        Random rng = new(seed);
        float[] data = new float[n];
        for (int i = 0; i < n; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }

    private static unsafe void AssertClose(float* a, float* b, int n, float rel)
    {
        float maxRel = 0;
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            float scale = MathF.Max(1e-3f, MathF.Max(MathF.Abs(a[i]), MathF.Abs(b[i])));
            float r = d / scale;
            if (r > maxRel)
                maxRel = r;
        }

        Assert.True(maxRel < rel, $"maxRel={maxRel}");
    }
}
