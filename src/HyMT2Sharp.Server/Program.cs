using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using System.Text.Json.Serialization;
using Sdcb.HyMT2Sharp.Model;
using Sdcb.HyMT2Sharp.Server;

string modelPath = GetArg(args, "--model", "-m")
    ?? Environment.GetEnvironmentVariable("HYMT2_MODEL")
    ?? @"D:\_\model\Hy-MT2-1.8B-2Bit.gguf";
int threads = GetInt(args, 0, "--threads", "-t");
int maxTokens = GetInt(args, 256, "--max-tokens");
// Stateless translation serving defaults to no cross-request cache; "disk"
// is reserved for a later block-store tier.
KvCachePolicy kvCache = GetArg(args, "--kv-cache")?.ToLowerInvariant() switch
{
    null or "none" => KvCachePolicy.None,
    "memory" => KvCachePolicy.Memory,
    string other => throw new ArgumentException($"--kv-cache must be none|memory, got {other}"),
};
string urls = GetArg(args, "--urls")
    ?? Environment.GetEnvironmentVariable("ASPNETCORE_URLS")
    ?? "http://127.0.0.1:8080";

Console.OutputEncoding = System.Text.Encoding.UTF8;
Console.WriteLine($"HyMT2Sharp.Server  model={modelPath}");
Console.WriteLine($"threads={(threads <= 0 ? "auto" : threads.ToString())}  avx2={Avx2.IsSupported}  vnni={AvxVnni.IsSupported}");

JsonSerializerOptions json = new(JsonSerializerOptions.Web);
ConfigureJson(json);

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);
builder.WebHost.UseUrls(urls);
builder.Services.ConfigureHttpJsonOptions(options => ConfigureJson(options.SerializerOptions));
builder.Services.AddCors(options => options.AddDefaultPolicy(policy =>
    policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod()));
builder.Services.AddSingleton(_ => new ChatCompletionService(modelPath, threads, maxTokens, kvCache));

WebApplication app = builder.Build();
app.UseCors();
app.UseDefaultFiles();
app.UseStaticFiles();

app.MapGet("/health", (ChatCompletionService svc) => Results.Json(new
{
    status = "ok",
    model = svc.ModelId,
    threads = svc.Model.ThreadCount,
    arch = svc.Model.Config.Architecture,
}));

bool debugBuild =
#if DEBUG
    true;
#else
    false;
#endif

app.MapGet("/props", (ChatCompletionService svc) => Results.Json(new
{
    model_path = svc.ModelPath,
    model_alias = svc.ModelId,
    default_generation_settings = new { n_ctx = svc.Model.Config.ContextLength, n_predict = svc.DefaultMaxTokens },
    total_slots = 1,
    n_threads = svc.Model.ThreadCount,
    debug = debugBuild,
    debugger_attached = Debugger.IsAttached,
}));

app.MapGet("/v1/models", (ChatCompletionService svc) => Results.Json(new
{
    @object = "list",
    data = new[]
    {
        new
        {
            id = svc.ModelId,
            @object = "model",
            created = svc.CreatedUnix,
            owned_by = "hymt2sharp",
            meta = new
            {
                n_vocab = svc.Model.Config.VocabSize,
                n_ctx_train = svc.Model.Config.ContextLength,
                n_embd = svc.Model.Config.HiddenSize,
                n_layer = svc.Model.Config.NumLayers,
            },
        },
    },
}));

app.MapPost("/v1/chat/completions", (ChatCompletionRequest request, ChatCompletionService svc, CancellationToken cancellationToken)
    => Complete(request, svc, json, cancellationToken));
app.MapPost("/chat/completions", (ChatCompletionRequest request, ChatCompletionService svc, CancellationToken cancellationToken)
    => Complete(request, svc, json, cancellationToken));

app.MapFallbackToFile("index.html");

_ = app.Services.GetRequiredService<ChatCompletionService>();
Console.WriteLine($"listening {urls}");
app.Run();

static async Task<IResult> Complete(
    ChatCompletionRequest request,
    ChatCompletionService svc,
    JsonSerializerOptions json,
    CancellationToken cancellationToken)
{
    if (request.Messages.Count == 0)
        return Results.BadRequest(new ErrorResponse { Error = new ErrorBody { Message = "messages is required" } });

    try
    {
        if (request.Stream)
            return new SseResult(svc.CompleteStreamAsync(request, cancellationToken), json);

        ChatCompletionResponse response = await svc.CompleteAsync(request, cancellationToken);
        return Results.Json(response, json);
    }
    catch (ArgumentException ex)
    {
        return Results.BadRequest(new ErrorResponse { Error = new ErrorBody { Message = ex.Message } });
    }
    catch (OperationCanceledException)
    {
        return Results.StatusCode(499);
    }
}

static void ConfigureJson(JsonSerializerOptions options)
{
    options.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
    options.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
    options.PropertyNameCaseInsensitive = true;
}

static string? GetArg(string[] args, params string[] names)
{
    for (int i = 0; i < args.Length; i++)
    {
        if (names.Contains(args[i], StringComparer.OrdinalIgnoreCase) && i + 1 < args.Length)
            return args[i + 1];
    }

    return null;
}

static int GetInt(string[] args, int fallback, params string[] names)
    => int.TryParse(GetArg(args, names), out int value) ? value : fallback;
