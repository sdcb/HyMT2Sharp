using System.Diagnostics;
using System.Text;
using Sdcb.HyMT2Sharp.Model;

string modelPath = Args.Get(args, "--model")
    ?? @"D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf";
string? prompt = Args.Get(args, "--prompt");
string? system = Args.Get(args, "--system");
int maxTokens = Args.GetInt(args, "--max-tokens", 256);
int threads = Args.GetInt(args, "--threads", 0);
Console.OutputEncoding = Encoding.UTF8;
Console.InputEncoding = Encoding.UTF8;

Console.WriteLine($"HyMT2Sharp  model={modelPath}");
Console.WriteLine($"threads={(threads <= 0 ? "auto" : threads.ToString())}  avx2={System.Runtime.Intrinsics.X86.Avx2.IsSupported}  vnni={System.Runtime.Intrinsics.X86.AvxVnni.IsSupported}");

IComputeBackend? backend = null;
string? backendName = Args.Get(args, "--backend");
if (backendName is not null and not "cpu")
{
    if (backendName == "metal" && System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX))
        backend = new Sdcb.HyMT2Sharp.Backends.Metal.MetalBackend();
    else
        Console.WriteLine($"backend {backendName} unavailable here — using cpu");
}
if (backend is not null)
    Console.WriteLine($"backend={backend.Name}");

using HunyuanDenseModel model = new(modelPath, threads, backend: backend);
Console.WriteLine($"threads={model.ThreadCount}{(threads <= 0 ? $" ({model.ThreadAutoHint})" : "")}  arch={model.Config.Architecture} layers={model.Config.NumLayers} hidden={model.Config.HiddenSize} heads={model.Config.NumHeads}/{model.Config.NumKvHeads} vocab={model.Config.VocabSize}");

List<ChatMessage> messages = [];
if (!string.IsNullOrEmpty(system))
    messages.Add(new ChatMessage("system", system));

if (!string.IsNullOrEmpty(prompt))
{
    Reply(model, messages, prompt, maxTokens);
    return;
}

if (!Console.IsInputRedirected)
{
    Console.WriteLine();
    Console.WriteLine("Type a message.  /reset  /exit");
}

while (true)
{
    if (!Console.IsInputRedirected)
        Console.Write("\nYou: ");
    string? line = Console.ReadLine();
    if (line == null)
        break;
    line = line.Trim();
    if (line.Length == 0)
        continue;
    if (line is "/exit" or "/quit")
        break;
    if (line is "/reset" or "/clear")
    {
        messages.Clear();
        if (!string.IsNullOrEmpty(system))
            messages.Add(new ChatMessage("system", system));
        model.ResetCache();
        Console.WriteLine("reset");
        continue;
    }

    if (line is "/help")
    {
        Console.WriteLine("/reset  clear conversation");
        Console.WriteLine("/exit   quit");
        continue;
    }

    Reply(model, messages, line, maxTokens);
}

static void Reply(HunyuanDenseModel model, List<ChatMessage> messages, string user, int maxTokens)
{
    messages.Add(new ChatMessage("user", user));
    string rendered = ChatTemplate.RenderHunyuanDense(messages);
    int[] promptIds = model.Tokenizer.Encode(rendered);
    Stopwatch gen = Stopwatch.StartNew();
    PromptAlignment align = model.AlignPrompt(promptIds);
    float[] logits = model.Forward(align.Suffix);
    double prefillMs = gen.Elapsed.TotalMilliseconds;
    int token = ArgMax(logits);
    List<int> generated = [];
    if (!model.Tokenizer.IsStop(token))
        generated.Add(token);

    if (!Console.IsOutputRedirected)
        Console.Write("Assistant: ");
    string visible = "";
    WriteDelta(model, generated, ref visible);

    gen.Restart();
    int decoded = 0;
    for (int i = 0; i < maxTokens; i++)
    {
        if (model.Tokenizer.IsStop(token))
            break;
        logits = model.Forward([token]);
        token = ArgMax(logits);
        decoded++;
        if (model.Tokenizer.IsStop(token))
            break;
        generated.Add(token);
        WriteDelta(model, generated, ref visible);
    }

    if (!Console.IsOutputRedirected)
        Console.WriteLine();

    double decodeMs = gen.Elapsed.TotalMilliseconds;
    string text = model.Tokenizer.DecodeVisible(generated);
    messages.Add(new ChatMessage("assistant", text));
    if (Console.IsOutputRedirected)
        Console.WriteLine(text);

    int processedPrefill = align.Suffix.Length;
    string prefillStats = $"prefill tokens={promptIds.Length} (cached={align.Cached})  {prefillMs:F1} ms  {processedPrefill / (prefillMs / 1000.0):F2} tok/s";
    if (decoded > 0)
        Console.WriteLine($"{prefillStats}  |  decode tokens={decoded}  {decodeMs:F1} ms  {decoded / (decodeMs / 1000.0):F2} tok/s");
    else
        Console.WriteLine($"{prefillStats}  |  decode tokens=0");
}

static void WriteDelta(HunyuanDenseModel model, List<int> generated, ref string visible)
{
    if (Console.IsOutputRedirected)
        return;
    string next = model.Tokenizer.DecodeVisible(generated);
    if (next.Length > visible.Length && next.StartsWith(visible, StringComparison.Ordinal))
        Console.Write(next[visible.Length..]);
    else if (next != visible)
        Console.Write(next);
    visible = next;
    Console.Out.Flush();
}

static int ArgMax(float[] logits) => Sampler.ArgMax(logits);

static class Args
{
    public static string? Get(string[] args, string name)
    {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }

    public static int GetInt(string[] args, string name, int fallback)
        => int.TryParse(Get(args, name), out int value) ? value : fallback;
}
