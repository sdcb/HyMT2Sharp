using System.Text;
using System.Text.RegularExpressions;
using Sdcb.HyMT2Sharp.Gguf;

namespace Sdcb.HyMT2Sharp.Model;

public sealed class BpeTokenizer
{
    private readonly string[] _idToToken;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<string, int> _merges;
    private readonly List<string> _specialTokens;
    private readonly Regex _pretokenizer;

    public int BosId { get; }
    public int EosId { get; }
    public int UnkId { get; }
    public int EogId { get; }
    public int VocabSize => _idToToken.Length;

    public BpeTokenizer(GgufFile gguf)
    {
        string[] tokens = gguf.GetStringArray("tokenizer.ggml.tokens");
        if (tokens.Length == 0)
            throw new InvalidDataException("GGUF missing tokenizer.ggml.tokens");

        _idToToken = tokens;
        _tokenToId = new Dictionary<string, int>(tokens.Length, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++)
            _tokenToId[tokens[i]] = i;

        _merges = new Dictionary<string, int>(StringComparer.Ordinal);
        string[] merges = gguf.GetStringArray("tokenizer.ggml.merges");
        for (int i = 0; i < merges.Length; i++)
            _merges.TryAdd(merges[i], i);

        BosId = gguf.GetInt32("tokenizer.ggml.bos_token_id", 0);
        EosId = gguf.GetInt32("tokenizer.ggml.eos_token_id", 0);
        UnkId = gguf.GetInt32("tokenizer.ggml.unknown_token_id", 0);
        EogId = _tokenToId.TryGetValue("<｜hy_place▁holder▁no▁2｜>", out int eog) ? eog : EosId;

        int[] tokenTypes = gguf.GetInt32Array("tokenizer.ggml.token_type");
        _specialTokens = [];
        for (int i = 0; i < tokens.Length; i++)
        {
            bool typed = i < tokenTypes.Length && (tokenTypes[i] == 3 || tokenTypes[i] == 4);
            if (typed || tokens[i].Contains("｜hy_", StringComparison.Ordinal) || i == BosId || i == EosId)
                _specialTokens.Add(tokens[i]);
        }

        _specialTokens.Sort((a, b) => b.Length.CompareTo(a.Length));
        string pre = gguf.GetString("tokenizer.ggml.pre", "gpt-2");
        _pretokenizer = new Regex(ResolvePreTokenizer(pre), RegexOptions.Compiled);
    }

    public int[] Encode(string text)
    {
        List<(string Text, int[]? Ids)> fragments = [(text, null)];
        foreach (string special in _specialTokens)
        {
            if (!_tokenToId.TryGetValue(special, out int specialId))
                continue;
            List<(string Text, int[]? Ids)> next = [];
            foreach ((string frag, int[]? presetIds) in fragments)
            {
                if (presetIds != null)
                {
                    next.Add((frag, presetIds));
                    continue;
                }

                int start = 0;
                while (true)
                {
                    int idx = frag.IndexOf(special, start, StringComparison.Ordinal);
                    if (idx < 0)
                    {
                        if (start < frag.Length)
                            next.Add((frag[start..], null));
                        break;
                    }

                    if (idx > start)
                        next.Add((frag[start..idx], null));
                    next.Add((special, [specialId]));
                    start = idx + special.Length;
                }
            }

            fragments = next;
        }

        List<int> ids = [];
        foreach ((string frag, int[]? preset) in fragments)
        {
            if (preset != null)
            {
                ids.AddRange(preset);
                continue;
            }

            foreach (Match match in _pretokenizer.Matches(frag))
            {
                string normalized = ByteEncode(match.Value);
                if (_tokenToId.TryGetValue(normalized, out int direct))
                {
                    ids.Add(direct);
                    continue;
                }

                ids.AddRange(Bpe(normalized));
            }
        }

        return [.. ids];
    }

    public bool IsStop(int id) => id == EosId || id == EogId;

    public bool TryLookup(string token, out int id) => _tokenToId.TryGetValue(token, out id);

    public string DecodeVisible(IReadOnlyList<int> ids)
    {
        List<int> kept = new(ids.Count);
        for (int i = 0; i < ids.Count; i++)
        {
            int id = ids[i];
            if ((uint)id >= (uint)_idToToken.Length)
                continue;
            if (_idToToken[id].Contains("｜hy_", StringComparison.Ordinal))
                continue;
            kept.Add(id);
        }

        return Decode(kept);
    }

    public string Decode(IReadOnlyList<int> ids)
    {
        List<byte> bytes = new(ids.Count);
        for (int i = 0; i < ids.Count; i++)
        {
            int id = ids[i];
            if ((uint)id >= (uint)_idToToken.Length)
                continue;
            AppendTokenBytes(_idToToken[id], bytes);
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    private List<int> Bpe(string token)
    {
        if (token.Length == 0)
            return [];
        List<string> word = new(token.Length);
        for (int i = 0; i < token.Length; i++)
            word.Add(token[i].ToString());

        while (word.Count > 1)
        {
            int best = int.MaxValue;
            int bestIndex = -1;
            for (int i = 0; i < word.Count - 1; i++)
            {
                if (_merges.TryGetValue(word[i] + " " + word[i + 1], out int rank) && rank < best)
                {
                    best = rank;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0)
                break;
            word[bestIndex] = word[bestIndex] + word[bestIndex + 1];
            word.RemoveAt(bestIndex + 1);
        }

        List<int> ids = new(word.Count);
        for (int i = 0; i < word.Count; i++)
        {
            if (_tokenToId.TryGetValue(word[i], out int id))
                ids.Add(id);
            else
                ids.Add(UnkId);
        }

        return ids;
    }

    internal static string ResolvePreTokenizer(string preTokenizerType) => preTokenizerType switch
    {
        "hunyuan-dense" or "deepseek-v3" or "joyai-llm" or "hy_v4" =>
            @"\p{N}{1,3}|" +
            @"[一-龥぀-ゟ゠-ヿ]+|" +
            @"[!""#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|" +
            @"[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|" +
            @" ?[\p{P}\p{S}]+[\r\n]*|" +
            @"\s*[\r\n]+|\s+(?!\S)|\s+",
        _ =>
            @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    };

    private static string ByteEncode(string text)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        char[] chars = new char[bytes.Length];
        for (int i = 0; i < bytes.Length; i++)
            chars[i] = EncodeByte(bytes[i]);
        return new string(chars);
    }

    private static char EncodeByte(byte b)
    {
        if (b == 0xAD)
            return (char)0x0143;
        if (b <= 0x20)
            return (char)(b + 0x0100);
        if (b >= 0x7F && b <= 0xA0)
            return (char)(b + 0x00A2);
        return (char)b;
    }

    private static void AppendTokenBytes(string token, List<byte> bytes)
    {
        foreach (char r in token)
        {
            if (r == 0x0100)
                continue;
            if (r > 0x0100 && r <= 0x0120)
            {
                bytes.Add((byte)(r - 0x0100));
                continue;
            }

            if (r >= 0x0121 && r <= 0x0142)
            {
                bytes.Add((byte)(r - 0x00A2));
                continue;
            }

            if (r == 0x0143)
            {
                bytes.Add(0xAD);
                continue;
            }

            if (r == 0x2581)
            {
                bytes.Add(0x20);
                continue;
            }

            if (r < 0x100)
            {
                bytes.Add((byte)r);
                continue;
            }

            bytes.AddRange(Encoding.UTF8.GetBytes([r]));
        }
    }
}
