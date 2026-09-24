using System.Diagnostics;
using System.Runtime.CompilerServices;
using Sdcb.HyMT2Sharp.Model;

namespace Sdcb.HyMT2Sharp.Server;

public sealed class ChatCompletionService : IDisposable
{
    private readonly HunyuanDenseModel _model;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly int _defaultMaxTokens;

    public ChatCompletionService(string modelPath, int threads, int defaultMaxTokens = 256)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model not found: {modelPath}", modelPath);

        ModelPath = Path.GetFullPath(modelPath);
        ModelId = Path.GetFileNameWithoutExtension(ModelPath);
        _defaultMaxTokens = defaultMaxTokens > 0 ? defaultMaxTokens : 256;
        CreatedUnix = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        Console.WriteLine($"loading {ModelPath}");
        _model = new HunyuanDenseModel(ModelPath, threads);
        Console.WriteLine($"ready  threads={_model.ThreadCount}{(threads <= 0 ? $" ({_model.ThreadAutoHint})" : "")}  arch={_model.Config.Architecture} layers={_model.Config.NumLayers} hidden={_model.Config.HiddenSize} heads={_model.Config.NumHeads}/{_model.Config.NumKvHeads} vocab={_model.Config.VocabSize}");
    }

    public string ModelId { get; }
    public string ModelPath { get; }
    public long CreatedUnix { get; }
    public HunyuanDenseModel Model => _model;
    public int DefaultMaxTokens => _defaultMaxTokens;

    public async Task<ChatCompletionResponse> CompleteAsync(ChatCompletionRequest request, CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken);
        bool ok = false;
        try
        {
            ChatCompletionResponse? response = null;
            foreach (object item in Generate(request, stream: false, cancellationToken))
            {
                if (item is ChatCompletionResponse done)
                    response = done;
            }

            if (response is null)
                throw new InvalidOperationException("generation produced no response");
            ok = true;
            return response;
        }
        finally
        {
            if (!ok)
                ResetSession();
            _gate.Release();
        }
    }

    public async IAsyncEnumerable<ChatCompletionChunk> CompleteStreamAsync(
        ChatCompletionRequest request,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken);
        bool ok = false;
        try
        {
            foreach (object item in Generate(request, stream: true, cancellationToken))
            {
                if (item is ChatCompletionChunk chunk)
                    yield return chunk;
            }

            ok = true;
        }
        finally
        {
            if (!ok)
                ResetSession();
            _gate.Release();
        }
    }

    private IEnumerable<object> Generate(ChatCompletionRequest request, bool stream, CancellationToken cancellationToken)
    {
        string id = "chatcmpl-" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        string model = string.IsNullOrWhiteSpace(request.Model) ? ModelId : request.Model!;
        int maxTokens = request.ResolveMaxTokens(_defaultMaxTokens);
        bool includeUsage = request.StreamOptions?.IncludeUsage ?? true;

        List<ChatMessage> messages = [];
        foreach (ChatCompletionMessageDto dto in request.Messages)
        {
            string role = dto.Role.Trim().ToLowerInvariant();
            if (role is not ("system" or "user" or "assistant"))
                continue;
            messages.Add(new ChatMessage(role, dto.GetText()));
        }

        if (messages.Count == 0)
            throw new ArgumentException("messages must contain at least one system, user, or assistant turn.");

        string rendered = ChatTemplate.RenderHunyuanDense(messages);
        int[] promptIds = _model.Tokenizer.Encode(rendered);
        if (promptIds.Length == 0)
            throw new ArgumentException("prompt produced no tokens.");
        if (promptIds.Length >= _model.Config.ContextLength)
            throw new ArgumentException($"prompt is {promptIds.Length} tokens, exceeding context {_model.Config.ContextLength}.");

        PromptAlignment align = _model.AlignPrompt(promptIds);
        Sampler sampler = new(request.ResolveSampling());
        sampler.FeedHistory(align.Suffix);
        Stopwatch timer = Stopwatch.StartNew();
        float[] logits = _model.Forward(align.Suffix);
        double promptMs = timer.Elapsed.TotalMilliseconds;

        if (stream)
        {
            yield return new ChatCompletionChunk
            {
                Id = id,
                Created = created,
                Model = model,
                Choices = [new ChatCompletionChunkChoice { Delta = new ChatCompletionDelta { Role = "assistant", Content = "" } }],
            };
        }

        int token = sampler.Sample(logits);
        List<int> generated = [];
        string visible = "";
        string finish = "length";
        timer.Restart();
        int decoded = 0;
        for (int i = 0; i < maxTokens; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (_model.Tokenizer.IsStop(token))
            {
                finish = "stop";
                break;
            }

            generated.Add(token);
            if (stream)
            {
                foreach (string delta in DrainDeltas(_model, generated, ref visible))
                    yield return ContentChunk(id, created, model, delta);
            }

            logits = _model.Forward([token]);
            token = sampler.Sample(logits);
            decoded++;
        }

        if (_model.Tokenizer.IsStop(token))
            finish = "stop";

        double decodeMs = timer.Elapsed.TotalMilliseconds;
        string text = _model.Tokenizer.DecodeVisible(generated);
        if (generated.Count >= maxTokens && finish != "stop")
            finish = "length";

        ChatCompletionUsage usage = new()
        {
            PromptTokens = promptIds.Length,
            CompletionTokens = generated.Count,
            TotalTokens = promptIds.Length + generated.Count,
        };
        ChatCompletionTimings timings = new()
        {
            PromptN = promptIds.Length,
            PromptCached = align.Cached,
            PromptMs = promptMs,
            PredictedN = decoded,
            PredictedMs = decodeMs,
            PredictedPerSecond = decoded > 0 && decodeMs > 0 ? decoded / (decodeMs / 1000.0) : 0,
        };

        ChatCompletionResponse response = new()
        {
            Id = id,
            Created = created,
            Model = model,
            Choices =
            [
                new ChatCompletionChoice
                {
                    Message = new ChatCompletionOutputMessage { Content = text },
                    FinishReason = finish,
                },
            ],
            Usage = usage,
            Timings = timings,
        };

        if (stream)
        {
            yield return new ChatCompletionChunk
            {
                Id = id,
                Created = created,
                Model = model,
                Choices = [new ChatCompletionChunkChoice { Delta = new ChatCompletionDelta(), FinishReason = finish }],
                Usage = includeUsage ? usage : null,
                Timings = timings,
            };
        }
        else
        {
            yield return response;
        }
    }

    private void ResetSession() => _model.ResetCache();

    private static ChatCompletionChunk ContentChunk(string id, long created, string model, string delta)
        => new()
        {
            Id = id,
            Created = created,
            Model = model,
            Choices = [new ChatCompletionChunkChoice { Delta = new ChatCompletionDelta { Content = delta } }],
        };

    private static List<string> DrainDeltas(HunyuanDenseModel model, List<int> generated, ref string visible)
    {
        string next = model.Tokenizer.DecodeVisible(generated);
        List<string> deltas = [];
        if (next.Length > visible.Length && next.StartsWith(visible, StringComparison.Ordinal))
        {
            string piece = next[visible.Length..];
            if (piece.Length > 0)
                deltas.Add(piece);
        }
        else if (next != visible)
        {
            deltas.Add(next);
        }

        visible = next;
        return deltas;
    }

    public void Dispose()
    {
        _gate.Dispose();
        _model.Dispose();
    }
}
