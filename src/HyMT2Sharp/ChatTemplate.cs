using System.Text;

namespace Sdcb.HyMT2Sharp.Model;

public readonly record struct ChatMessage(string Role, string Content);

public static class ChatTemplate
{
    /// <summary>
    /// Official Hy-MT2 Jinja: always opens with BOS. The assistant marker is
    /// appended only when <paramref name="addGenerationPrompt"/> is set.
    /// </summary>
    public static string RenderHunyuanDense(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt = true)
    {
        StringBuilder sb = new();
        int startIdx = 0;
        if (messages.Count > 0 && messages[0].Role == "system")
        {
            sb.Append("<｜hy_begin▁of▁sentence｜>");
            sb.Append(messages[0].Content);
            sb.Append("<｜hy_place▁holder▁no▁3｜>");
            startIdx = 1;
        }
        else
        {
            sb.Append("<｜hy_begin▁of▁sentence｜>");
        }

        for (int i = startIdx; i < messages.Count; i++)
        {
            ChatMessage msg = messages[i];
            if (msg.Role == "user")
            {
                sb.Append("<｜hy_User｜>");
                sb.Append(msg.Content);
            }
            else if (msg.Role == "assistant")
            {
                sb.Append("<｜hy_Assistant｜>");
                sb.Append(msg.Content);
                sb.Append("<｜hy_place▁holder▁no▁2｜>");
            }
        }

        if (addGenerationPrompt)
            sb.Append("<｜hy_Assistant｜>");
        else
            sb.Append("<｜hy_place▁holder▁no▁8｜>");

        return sb.ToString();
    }
}
