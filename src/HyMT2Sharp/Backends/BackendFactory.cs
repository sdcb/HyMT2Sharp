using System.Runtime.InteropServices;
using Sdcb.HyMT2Sharp.Model;

namespace Sdcb.HyMT2Sharp.Backends;

/// <summary>
/// Backend selection: an explicit name constructs that backend (throwing if it
/// cannot run); "auto"/null probes each known backend in preference order and
/// returns the first that reports a usable device — null means plain CPU.
/// </summary>
public static class BackendFactory
{
    /// <param name="preference">"auto"/null, or "cpu" | "metal" | "vulkan".
    /// Falls back to the HYMT_BACKEND environment variable when null.</param>
    public static IComputeBackend? Create(string? preference = null)
    {
        string p = (preference ?? Environment.GetEnvironmentVariable("HYMT_BACKEND") ?? "auto")
            .Trim().ToLowerInvariant();
        return p switch
        {
            "" or "auto" => CreateAuto(),
            "cpu" => null,
            "metal" => new Metal.MetalBackend(),
            "vulkan" => new Vulkan.VulkanBackend(),
            _ => throw new ArgumentException($"unknown backend '{p}' (expected auto|cpu|metal|vulkan)"),
        };
    }

    private static IComputeBackend? CreateAuto()
    {
        // Probe order: platform-native first, then Vulkan, then CPU (null).
        // Each probe is cheap — device enumeration only, no model upload.
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX) && TryCreate(() => new Metal.MetalBackend(), out var m))
            return m;
        if (Vulkan.VulkanBackend.IsSupported() && TryCreate(() => new Vulkan.VulkanBackend(), out var v))
            return v;
        return null;
    }

    private static bool TryCreate(Func<IComputeBackend> f, out IComputeBackend backend)
    {
        try { backend = f(); return true; }
        catch { backend = null!; return false; }
    }
}
