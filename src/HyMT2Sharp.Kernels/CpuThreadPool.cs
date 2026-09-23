using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Dedicated workers pinned to physical P-cores when the OS exposes topology.
/// Workers poll a job id like ggml, then park on an event so sequential
/// RMS/SiLU does not burn the P-cores. <see cref="Barrier"/> is the ggml
/// spin barrier for quantize→GEMM inside one <see cref="For"/>.
/// </summary>
public sealed class CpuThreadPool : IDisposable
{
    private readonly Thread[] _threads;
    private readonly AutoResetEvent[] _starts;
    private readonly int[] _parked;
    private Action<int, int>? _work;
    private int _workers;
    private int _remaining;
    private int _barrierCount;
    private int _barrierGen;
    private int _jobId;
    private volatile bool _stop;

    public int ThreadCount { get; }

    public CpuThreadPool(int threadCount = 0)
    {
        if (threadCount <= 0)
            threadCount = CpuTopology.PreferPCoreCount;
        ThreadCount = Math.Max(1, threadCount);
        _threads = new Thread[ThreadCount];
        _starts = new AutoResetEvent[ThreadCount];
        _parked = new int[ThreadCount];
        for (int i = 0; i < ThreadCount; i++)
        {
            int index = i;
            _starts[i] = new AutoResetEvent(false);
            _threads[i] = new Thread(() => Worker(index))
            {
                IsBackground = true,
                Name = $"hymt2-cpu-{index}",
                Priority = ThreadPriority.AboveNormal,
            };
            _threads[i].Start();
        }
    }

    public void For(int count, Action<int, int> body)
    {
        if (count <= 1 || ThreadCount == 1)
        {
            body(0, 1);
            return;
        }

        int workers = Math.Min(ThreadCount, count);
        _work = body;
        Volatile.Write(ref _workers, workers);
        Volatile.Write(ref _remaining, workers);
        Interlocked.Increment(ref _jobId);
        for (int i = 0; i < workers; i++)
            if (Volatile.Read(ref _parked[i]) != 0)
                _starts[i].Set();

        int spins = 0;
        while (Volatile.Read(ref _remaining) > 0)
        {
            Thread.SpinWait(64);
            if (++spins == 4096)
            {
                Thread.Yield();
                spins = 0;
            }
        }

        _work = null;
    }

    public void Barrier()
    {
        int n = Volatile.Read(ref _workers);
        if (n <= 1)
            return;

        int gen = Volatile.Read(ref _barrierGen);
        if (Interlocked.Increment(ref _barrierCount) == n)
        {
            Volatile.Write(ref _barrierCount, 0);
            Interlocked.Increment(ref _barrierGen);
            return;
        }

        while (Volatile.Read(ref _barrierGen) == gen)
            Thread.SpinWait(32);
    }

    private void Worker(int index)
    {
        TryPinToPCore(index);
        int seen = 0;
        while (true)
        {
            int job = Volatile.Read(ref _jobId);
            if (job != seen)
            {
                seen = job;
                if (_stop)
                    return;
                int workers = Volatile.Read(ref _workers);
                if (index < workers)
                {
                    Action<int, int>? work = _work;
                    work?.Invoke(index, workers);
                    Interlocked.Decrement(ref _remaining);
                }

                continue;
            }

            if (_stop)
                return;

            int spins = 0;
            while (Volatile.Read(ref _jobId) == seen && !_stop && spins < 8192)
            {
                Thread.SpinWait(64);
                spins++;
            }

            if (Volatile.Read(ref _jobId) != seen || _stop)
                continue;
            Interlocked.Exchange(ref _parked[index], 1);
            if (Volatile.Read(ref _jobId) == seen && !_stop)
                _starts[index].WaitOne();
            Volatile.Write(ref _parked[index], 0);
        }
    }

    public static int PreferPCoreCount() => CpuTopology.PreferPCoreCount;

    private void TryPinToPCore(int index)
    {
        if (!OperatingSystem.IsWindows())
            return;
        try
        {
            IReadOnlyList<int> targets = ThreadCount <= CpuTopology.PCoreLeaders.Count
                ? CpuTopology.PCoreLeaders
                : ThreadCount <= CpuTopology.PhysicalLeaders.Count
                    ? CpuTopology.PhysicalLeaders
                    : CpuTopology.PhysicalLogicalIds;
            if (targets.Count == 0)
                return;

            int cpu = targets[index % targets.Count];
            if ((uint)cpu >= 64)
                return;

            nint mask = (nint)(1UL << cpu);
            SetThreadAffinityMask(GetCurrentThread(), mask);
        }
        catch
        {
            // Affinity is best-effort.
        }
    }

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentThread();

    [DllImport("kernel32.dll")]
    private static extern IntPtr SetThreadAffinityMask(IntPtr thread, IntPtr mask);

    public void Dispose()
    {
        _stop = true;
        Interlocked.Increment(ref _jobId);
        for (int i = 0; i < ThreadCount; i++)
            _starts[i].Set();
        for (int i = 0; i < ThreadCount; i++)
            _threads[i].Join();
        for (int i = 0; i < ThreadCount; i++)
            _starts[i].Dispose();
    }
}
