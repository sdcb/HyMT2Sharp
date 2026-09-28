using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using Sdcb.HyMT2Sharp.Backends.Vulkan;
using Sdcb.HyMT2Sharp.Model;
using Bit125 = Sdcb.GgufWeights.Hy_MT2_1_8B_1_25Bit.Weights;
using Q4 = Sdcb.GgufWeights.Hy_MT2_1_8B_Q4_K_M.Weights;

Console.OutputEncoding = System.Text.Encoding.UTF8;

if (args.Length > 0 && args[0] == "child")
    return RunChild(args);

string projectDir = FindProjectDir();
string packages = Path.Combine(projectDir, "packages");
VerifyPackages(packages);
VerifyOutputDlls(AppContext.BaseDirectory);
VerifyStream("1.25Bit", Bit125.Length, Bit125.Sha256, Bit125.Segments().Length, 2, Bit125.Model);
VerifyStream("Q4_K_M", Q4.Length, Q4.Sha256, Q4.Segments().Length, 5, Q4.Model);

(string Model, string Backend)[] cases =
[
    ("1.25", "cpu"),
    ("q4", "cpu"),
    ("1.25", "vulkan"),
    ("q4", "vulkan"),
];

List<RunRow> rows = [];
foreach ((string model, string backend) in cases)
{
    if (!File.Exists(ModelPath(model)))
    {
        Console.WriteLine($"skip {backend} {model}: {ModelPath(model)} not found");
        continue;
    }

    RunRow path = await RunChildAsync(model, backend, "path");
    RunRow stream = await RunChildAsync(model, backend, "stream");
    bool same = path.Tokens == stream.Tokens;
    rows.Add(path);
    rows.Add(stream);
    Console.WriteLine($"{backend} {model} tokens {(same ? "match" : "DIFFER")}  n={path.Tokens.Split(',', StringSplitOptions.RemoveEmptyEntries).Length}");
    if (!same)
    {
        Console.WriteLine($"  path   {path.Tokens}");
        Console.WriteLine($"  stream {stream.Tokens}");
        return 1;
    }

    Console.WriteLine($"  text {path.Text}");
}

Console.WriteLine();
Console.WriteLine($"{"backend",-8} {"model",-6} {"source",-7} {"load_s",10} {"peak_MiB",10}");
foreach (RunRow row in rows)
    Console.WriteLine($"{row.Backend,-8} {row.Model,-6} {row.Source,-7} {row.LoadSeconds,10:F2} {row.PeakMiB,10:F1}");
return 0;

static void VerifyPackages(string packages)
{
    if (!Directory.Exists(packages))
        throw new DirectoryNotFoundException($"isolated package folder missing: {packages}");
    string[] dirs = Directory.GetDirectories(packages).Select(Path.GetFileName).ToArray()!;
    string[] required =
    [
        "sdcb.ggufweights.abstractions",
        "sdcb.ggufweights.hy-mt2-1.8b-1.25bit",
        "sdcb.ggufweights.hy-mt2-1.8b-1.25bit.part2",
        "sdcb.ggufweights.hy-mt2-1.8b-q4_k_m",
        "sdcb.ggufweights.hy-mt2-1.8b-q4_k_m.part2",
        "sdcb.ggufweights.hy-mt2-1.8b-q4_k_m.part3",
        "sdcb.ggufweights.hy-mt2-1.8b-q4_k_m.part4",
        "sdcb.ggufweights.hy-mt2-1.8b-q4_k_m.part5",
    ];
    foreach (string id in required)
    {
        if (!dirs.Any(d => string.Equals(d, id, StringComparison.OrdinalIgnoreCase)))
            throw new InvalidOperationException($"package folder missing {id} under {packages}");
    }

    Console.WriteLine($"packages ok ({dirs.Length} folders, parts came in as dependencies)");
}

static void VerifyOutputDlls(string outputDir)
{
    string[] required =
    [
        "Sdcb.GgufWeights.Abstractions.dll",
        "Sdcb.GgufWeights.Hy-MT2-1.8B-1.25Bit.dll",
        "Sdcb.GgufWeights.Hy-MT2-1.8B-1.25Bit.Part2.dll",
        "Sdcb.GgufWeights.Hy-MT2-1.8B-Q4_K_M.dll",
        "Sdcb.GgufWeights.Hy-MT2-1.8B-Q4_K_M.Part2.dll",
        "Sdcb.GgufWeights.Hy-MT2-1.8B-Q4_K_M.Part3.dll",
        "Sdcb.GgufWeights.Hy-MT2-1.8B-Q4_K_M.Part4.dll",
        "Sdcb.GgufWeights.Hy-MT2-1.8B-Q4_K_M.Part5.dll",
    ];
    foreach (string name in required)
    {
        string path = Path.Combine(outputDir, name);
        if (!File.Exists(path))
            throw new FileNotFoundException($"output missing {name}", path);
    }

    Console.WriteLine($"output dlls ok ({required.Length})");
}

static void VerifyStream(string label, long length, string sha256, int segments, int expectedSegments, Func<Stream> open)
{
    if (segments != expectedSegments)
        throw new InvalidOperationException($"{label}: Segments() = {segments}, expected {expectedSegments}");
    using Stream stream = open();
    if (stream.Length != length)
        throw new InvalidOperationException($"{label}: stream length {stream.Length} != Weights.Length {length}");
    byte[] magic = new byte[4];
    Fill(stream, magic);
    if (magic is not [0x47, 0x47, 0x55, 0x46])
        throw new InvalidOperationException($"{label}: magic is {Convert.ToHexString(magic)}");
    using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    hash.AppendData(magic);
    byte[] buf = new byte[1 << 20];
    while (true)
    {
        int n = stream.Read(buf, 0, buf.Length);
        if (n == 0)
            break;
        hash.AppendData(buf, 0, n);
    }

    string got = Convert.ToHexStringLower(hash.GetHashAndReset());
    if (!string.Equals(got, sha256, StringComparison.OrdinalIgnoreCase))
        throw new InvalidOperationException($"{label}: sha256 {got} != {sha256}");
    Console.WriteLine($"{label} stream ok  length={length:N0}  sha256={got}  segments={segments}");
}

static async Task<RunRow> RunChildAsync(string model, string backend, string source)
{
    string exe = Environment.ProcessPath ?? throw new InvalidOperationException("no process path");
    ProcessStartInfo psi = new()
    {
        FileName = exe,
        Arguments = $"child --model {model} --backend {backend} --source {source} --max-tokens 32",
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        WorkingDirectory = AppContext.BaseDirectory,
    };
    using Process proc = Process.Start(psi) ?? throw new InvalidOperationException("failed to start child");
    Task<string> stdoutTask = proc.StandardOutput.ReadToEndAsync();
    Task<string> stderrTask = proc.StandardError.ReadToEndAsync();
    await proc.WaitForExitAsync();
    string stdout = await stdoutTask;
    string stderr = await stderrTask;
    if (proc.ExitCode != 0)
        throw new InvalidOperationException($"child {backend} {model} {source} exited {proc.ExitCode}\n{stdout}\n{stderr}");

    string load = Required(stdout, "LOAD_MS");
    string peak = Required(stdout, "PEAK_BYTES");
    string tokens = Required(stdout, "TOKENS");
    string text = Required(stdout, "TEXT");
    double loadMs = double.Parse(load, CultureInfo.InvariantCulture);
    long peakBytes = long.Parse(peak, CultureInfo.InvariantCulture);
    Console.WriteLine($"  {backend} {model} {source}  load {loadMs / 1000.0:F2}s  peak {peakBytes / (1024.0 * 1024.0):F1} MiB");
    return new RunRow(backend, model, source, loadMs / 1000.0, peakBytes / (1024.0 * 1024.0), tokens, text);
}

static int RunChild(string[] args)
{
    string model = RequiredArg(args, "--model");
    string backendName = RequiredArg(args, "--backend");
    string source = RequiredArg(args, "--source");
    int maxTokens = int.Parse(RequiredArg(args, "--max-tokens"), CultureInfo.InvariantCulture);
    string path = ModelPath(model);

    IComputeBackend? backend = backendName == "vulkan" ? new VulkanBackend() : null;
    Stopwatch sw = Stopwatch.StartNew();
    using HunyuanDenseModel loaded = source == "stream"
        ? new HunyuanDenseModel(OpenWeights(model), threads: 8, backend: backend, leaveOpen: false)
        : new HunyuanDenseModel(path, threads: 8, backend: backend);
    sw.Stop();
    Process proc = Process.GetCurrentProcess();
    proc.Refresh();
    long peak = proc.PeakWorkingSet64;

    List<ChatMessage> messages =
    [
        new("system", "Translate the following text into English. Do not add explanations."),
        new("user", "今天天气很好，我们去公园走走吧。"),
    ];
    string rendered = ChatTemplate.RenderHunyuanDense(messages);
    int[] promptIds = loaded.Tokenizer.Encode(rendered);
    PromptAlignment align = loaded.AlignPrompt(promptIds);
    float[] logits = loaded.Forward(align.Suffix);
    int token = Sampler.ArgMax(logits);
    List<int> generated = [];
    if (!loaded.Tokenizer.IsStop(token))
        generated.Add(token);
    for (int i = 0; i < maxTokens && !loaded.Tokenizer.IsStop(token); i++)
    {
        logits = loaded.Forward([token]);
        token = Sampler.ArgMax(logits);
        if (loaded.Tokenizer.IsStop(token))
            break;
        generated.Add(token);
    }

    string text = loaded.Tokenizer.DecodeVisible(generated).Replace('\r', ' ').Replace('\n', ' ');
    Console.WriteLine($"LOAD_MS={sw.Elapsed.TotalMilliseconds.ToString("F1", CultureInfo.InvariantCulture)}");
    Console.WriteLine($"PEAK_BYTES={peak.ToString(CultureInfo.InvariantCulture)}");
    Console.WriteLine($"TOKENS={string.Join(',', generated)}");
    Console.WriteLine($"TEXT={text}");
    return 0;
}

static Stream OpenWeights(string model) => model switch
{
    "1.25" => Bit125.Model(),
    "q4" => Q4.Model(),
    _ => throw new ArgumentException(model),
};

static string ModelPath(string model) => model switch
{
    "1.25" => @"D:\_\model\Hy-MT2-1.8B-1.25Bit.gguf",
    "q4" => @"D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf",
    _ => throw new ArgumentException(model),
};

static string FindProjectDir()
{
    DirectoryInfo? dir = new(AppContext.BaseDirectory);
    while (dir is not null)
    {
        if (File.Exists(Path.Combine(dir.FullName, "EmbeddedWeightsDemo.csproj")))
            return dir.FullName;
        dir = dir.Parent;
    }

    throw new InvalidOperationException("EmbeddedWeightsDemo.csproj not found from " + AppContext.BaseDirectory);
}

static string Required(string text, string key)
{
    foreach (string line in text.Split('\n'))
    {
        string trimmed = line.Trim();
        if (trimmed.StartsWith(key + "=", StringComparison.Ordinal))
            return trimmed[(key.Length + 1)..];
    }

    throw new InvalidOperationException($"child output missing {key}\n{text}");
}

static string RequiredArg(string[] args, string name)
{
    int i = Array.IndexOf(args, name);
    if (i < 0 || i + 1 >= args.Length)
        throw new ArgumentException($"missing {name}");
    return args[i + 1];
}

static void Fill(Stream stream, byte[] buf)
{
    int got = 0;
    while (got < buf.Length)
    {
        int n = stream.Read(buf, got, buf.Length - got);
        if (n == 0)
            throw new EndOfStreamException("short stream while reading magic");
        got += n;
    }
}

internal sealed record RunRow(string Backend, string Model, string Source, double LoadSeconds, double PeakMiB, string Tokens, string Text);
