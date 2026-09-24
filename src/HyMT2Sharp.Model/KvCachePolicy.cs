namespace Sdcb.HyMT2Sharp.Model;

/// <summary>
/// Cross-request KV cache retention policy. <see cref="None"/> releases the
/// KV buffers when a request ends (stateless translation serving, where the
/// next prompt almost never shares a prefix); <see cref="Memory"/> keeps the
/// cache so the next prompt can reuse its longest matching prefix. A disk
/// block tier is reserved for a later change.
/// </summary>
public enum KvCachePolicy
{
    None,
    Memory,
}
