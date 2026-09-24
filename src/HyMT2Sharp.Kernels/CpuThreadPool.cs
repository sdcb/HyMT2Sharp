using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Dedicated workers pinned to physical P-cores when the OS exposes topology.
/// Workers poll a job id like ggml, then park on an event so sequential
/// RMS/SiLU does not burn the P-cores. <see cref="Barrier"/> is the ggml
/// spin barrier for quantize→GEMM inside one <see cref="For"/>.
/// The calling thread participates as worker 0 (ggml-style), so
/// <see cref="ThreadCount"/> counts total participants — running at
/// vCPU count does not oversubscribe a VM the way a spinning join would.
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
            threadCount = CalibrateThreadCount(CpuTopology.LogicalCount);
        ThreadCount = Math.Max(1, threadCount);
        _threads = new Thread[ThreadCount];
        _starts = new AutoResetEvent[ThreadCount];
        _parked = new int[ThreadCount];
        for (int i = 1; i < ThreadCount; i++)
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
        Volatile.Write(ref _remaining, workers - 1);
        Interlocked.Increment(ref _jobId);
        for (int i = 1; i < workers; i++)
            if (Volatile.Read(ref _parked[i]) != 0)
                _starts[i].Set();

        // Main participates as worker 0: it produces real work instead of
        // pure spinning, and the runnable set stays at ThreadCount instead
        // of ThreadCount + a spin-waiter (the 8-vCPU cliff).
        try
        {
            body(0, workers);
        }
        finally
        {
            int spins = 0;
            int yields = 0;
            while (Volatile.Read(ref _remaining) > 0)
            {
                Thread.SpinWait(64);
                if (++spins == 4096)
                {
                    Thread.Yield();
                    spins = 0;
                    // Yielding alone can't help when a whole vCPU is stolen — the
                    // guest just reschedules the spinner and the pCPU stays
                    // oversubscribed. Sleeping idles the vCPU so the hypervisor
                    // can schedule the stalled sibling. Only reached after ~0.5ms
                    // of sustained stall, so dedicated hardware never pays for it.
                    if (++yields % 8 == 0)
                        Thread.Sleep(1);
                }
            }

            _work = null;
        }
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

        // Fast path is pure spin (peers arrive within microseconds on dedicated
        // hardware). Only after a peer is genuinely preempted (~0.3ms+) do we
        // yield, and after sustained stalls we sleep: yield alone reschedules
        // guest threads while the vCPU stays busy, whereas an idle vCPU lets
        // the hypervisor run the stolen sibling — this is what collapses
        // decode on oversubscribed VMs and containers.
        int spins = 0;
        int yields = 0;
        while (Volatile.Read(ref _barrierGen) == gen)
        {
            Thread.SpinWait(32);
            if (++spins >= 4096)
            {
                Thread.Yield();
                spins = 0;
                if (++yields % 8 == 0)
                    Thread.Sleep(1);
            }
        }
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

    /// <summary>
    /// Measured worker count for <c>threads = 0</c>: probes barrier-heavy rounds
    /// (the decode regime) at every size up to <paramref name="maxThreads"/> and
    /// keeps the fastest, preferring fewer workers inside 3%. A noisy host can
    /// unfairly penalize large probes, so the result is floored at
    /// logical/4. Result is memoized per process;
    /// <c>HYMT2SHARP_CALIBRATION=0</c> falls back to
    /// <see cref="CpuTopology.PreferPCoreCount"/>. Costs a few hundred ms once.
    /// </summary>
    public static int CalibrateThreadCount(int maxThreads)
    {
        int max = Math.Max(1, maxThreads);
        if (max <= 2)
            return max;
        if (_calibrated > 0)
            return Math.Min(_calibrated, max);
        lock (_calibrateLock)
        {
            if (_calibrated < 0)
            {
                _calibrated = Environment.GetEnvironmentVariable("HYMT2SHARP_CALIBRATION") == "0"
                    ? max
                    : Math.Max(Probe(max), Math.Min(max, Math.Max(1, CpuTopology.LogicalCount / 4)));
            }
            return Math.Min(_calibrated, max);
        }
    }

    /// <summary>Best measured worker count from the last calibration (0 = not run).</summary>
    public static int CalibratedWorkers => _calibrated;

    private static readonly object _calibrateLock = new();
    private static readonly float[] _probeBuf = new float[65536];
    private static int _calibrated = -1;
    private static float _probeSink;

    /// <summary>
    /// Each probe round splits a fixed amount of work across workers via an
    /// interlocked cursor (the decode regime), with one explicit barrier between
    /// two phases (the prefill regime). More workers win only while the added
    /// parallelism outweighs dispatch and straggler cost — on oversubscribed
    /// VMs the rate drops once a vCPU gets preempted mid-round.
    /// </summary>
    private static int Probe(int max)
    {
        const int units = 24;
        // Two interleaved sweeps keep a transient noise burst from permanently
        // penalizing whichever size happens to be measured during it: each size
        // keeps its best rate across sweeps.
        double[] rates = new double[max + 1];
        for (int sweep = 0; sweep < 2; sweep++)
        for (int t = 1; t <= max; t++)
        {
            using CpuThreadPool pool = new(t);
            for (int i = 0; i < 16; i++)
                ProbeRound(pool, units);
            long start = Stopwatch.GetTimestamp();
            int iters = 0;
            while (iters < 120 && Stopwatch.GetElapsedTime(start).TotalMilliseconds < 30)
            {
                ProbeRound(pool, units);
                iters++;
            }
            double rate = iters / Stopwatch.GetElapsedTime(start).TotalSeconds;
            if (rate > rates[t])
                rates[t] = rate;
        }

        if (Environment.GetEnvironmentVariable("HYMT2SHARP_PROBE_DEBUG") == "1")
            Console.Error.WriteLine($"hymt2-probe rates(rounds/s): {string.Join(" ", Enumerable.Range(1, max).Select(t => $"t{t}={rates[t]:F0}"))}");

        int best = 1;
        double bestRate = 0;
        for (int t = 1; t <= max; t++)
        if (rates[t] > bestRate * 1.03)
        {
            bestRate = rates[t];
            best = t;
        }
        return best;
    }

    private static int _probeCursor;

    private static void ProbeRound(CpuThreadPool pool, int units)
    {
        _probeCursor = 0;
        pool.For(units, (_, _) =>
        {
            while (Interlocked.Increment(ref _probeCursor) <= units)
                ProbeWork();
            pool.Barrier();
            while (Interlocked.Increment(ref _probeCursor) <= units * 2)
                ProbeWork();
        });
    }

    private static void ProbeWork()
    {
        // Mixed unit: a 256KB streaming sum (the decode bandwidth regime) plus
        // a dependent-FMA chain (the compute regime). Pure-ALU probes miss the
        // contention that makes the last workers worthless on saturated VMs.
        float[] buf = _probeBuf;
        float s = 0;
        for (int i = 0; i < buf.Length; i++)
            s += buf[i];
        float x = _probeSink;
        for (int i = 0; i < 16384; i++)
            x = x * 1.0000001f + 0.25f;
        _probeSink = x + s;
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
        for (int i = 1; i < ThreadCount; i++)
            _starts[i].Set();
        for (int i = 1; i < ThreadCount; i++)
            _threads[i].Join();
        for (int i = 1; i < ThreadCount; i++)
            _starts[i].Dispose();
    }
}
