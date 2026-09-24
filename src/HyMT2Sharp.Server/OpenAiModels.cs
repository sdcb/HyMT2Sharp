using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Sdcb.HyMT2Sharp.Model;

namespace Sdcb.HyMT2Sharp.Server;

public sealed class ChatCompletionRequest
{
    public string? Model { get; set; }
    public List<ChatCompletionMessageDto> Messages { get; set; } = [];
    public int? MaxTokens { get; set; }
    public int? MaxCompletionTokens { get; set; }
    public bool Stream { get; set; }
    public StreamOptions? StreamOptions { get; set; }

    // Sampling. Null temperature keeps historical behavior: greedy argmax.
    public float? Temperature { get; set; }
    public float? TopP { get; set; }
    public int? TopK { get; set; }
    public float? MinP { get; set; }
    public float? RepetitionPenalty { get; set; }
    public int? Seed { get; set; }

    public SamplingParams ResolveSampling()
    {
        return new SamplingParams
        {
            Temperature = Temperature ?? 0f,
            TopP = TopP ?? 1f,
            TopK = TopK ?? 0,
            MinP = MinP ?? 0f,
            RepeatPenalty = RepetitionPenalty ?? 1f,
            Seed = Seed,
        };
    }

    public int ResolveMaxTokens(int fallback)
    {
        if (MaxCompletionTokens is > 0)
            return MaxCompletionTokens.Value;
        if (MaxTokens is > 0)
            return MaxTokens.Value;
        return fallback;
    }
}

public sealed class StreamOptions
{
    public bool IncludeUsage { get; set; }
}

public sealed class ChatCompletionMessageDto
{
    public string Role { get; set; } = "";
    public JsonElement Content { get; set; }

    public string GetText()
    {
        if (Content.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null)
            return "";
        if (Content.ValueKind == JsonValueKind.String)
            return Content.GetString() ?? "";
        if (Content.ValueKind != JsonValueKind.Array)
            return Content.ToString();

        StringBuilder sb = new();
        foreach (JsonElement part in Content.EnumerateArray())
        {
            if (part.ValueKind == JsonValueKind.String)
            {
                sb.Append(part.GetString());
                continue;
            }

            if (part.ValueKind != JsonValueKind.Object)
                continue;
            if (part.TryGetProperty("text", out JsonElement text) && text.ValueKind == JsonValueKind.String)
                sb.Append(text.GetString());
        }

        return sb.ToString();
    }
}

public sealed class ChatCompletionResponse
{
    public required string Id { get; set; }
    public string Object { get; set; } = "chat.completion";
    public long Created { get; set; }
    public required string Model { get; set; }
    public required List<ChatCompletionChoice> Choices { get; set; }
    public ChatCompletionUsage? Usage { get; set; }
    public ChatCompletionTimings? Timings { get; set; }
}

public sealed class ChatCompletionChoice
{
    public int Index { get; set; }
    public ChatCompletionOutputMessage? Message { get; set; }
    public string? FinishReason { get; set; }
}

public sealed class ChatCompletionOutputMessage
{
    public string Role { get; set; } = "assistant";
    public string Content { get; set; } = "";
}

public sealed class ChatCompletionChunk
{
    public required string Id { get; set; }
    public string Object { get; set; } = "chat.completion.chunk";
    public long Created { get; set; }
    public required string Model { get; set; }
    public required List<ChatCompletionChunkChoice> Choices { get; set; }
    public ChatCompletionUsage? Usage { get; set; }
    public ChatCompletionTimings? Timings { get; set; }
}

public sealed class ChatCompletionChunkChoice
{
    public int Index { get; set; }
    public ChatCompletionDelta Delta { get; set; } = new();
    public string? FinishReason { get; set; }
}

public sealed class ChatCompletionDelta
{
    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Role { get; set; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Content { get; set; }
}

public sealed class ChatCompletionUsage
{
    public int PromptTokens { get; set; }
    public int CompletionTokens { get; set; }
    public int TotalTokens { get; set; }
}

public sealed class ChatCompletionTimings
{
    public int PromptN { get; set; }
    public int PromptCached { get; set; }
    public double PromptMs { get; set; }
    public int PredictedN { get; set; }
    public double PredictedMs { get; set; }
    public double PredictedPerSecond { get; set; }
}

public sealed class ErrorResponse
{
    public required ErrorBody Error { get; set; }
}

public sealed class ErrorBody
{
    public required string Message { get; set; }
    public string Type { get; set; } = "invalid_request_error";
    public string? Code { get; set; }
}
