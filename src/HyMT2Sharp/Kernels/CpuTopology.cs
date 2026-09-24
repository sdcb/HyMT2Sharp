using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Versioning;
using System.Text;
using Microsoft.Win32;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Discovers physical cores from the OS topology instead of guessing from
/// <see cref="Environment.ProcessorCount"/>. The default worker count is the
/// number of physical P-cores, not SMT siblings and not E-cores, so lockstep
/// GEMM is not held by slower cores.
/// </summary>
public static class CpuTopology
{
    public static IReadOnlyList<int> PCoreLogicalIds { get; }
    public static IReadOnlyList<int> PCoreLeaders { get; }
    public static IReadOnlyList<int> PhysicalLeaders { get; }
    public static IReadOnlyList<int> PhysicalLogicalIds { get; }
    public static int PreferPCoreCount { get; }
    public static int PhysicalCoreCount { get; }
    public static int LogicalCount { get; }
    public static bool SawSmt { get; }
    public static bool IsVirtualMachine { get; }
    public static string AutoHint { get; }

    static CpuTopology()
    {
        int logical = Math.Max(1, Environment.ProcessorCount);
        LogicalCount = logical;
        IsVirtualMachine = DetectVirtualMachine();

        List<Core> cores = [];
        try
        {
            if (OperatingSystem.IsWindows())
                TryQueryWindows(cores);
            else if (OperatingSystem.IsLinux())
                TryQueryLinux(cores);
        }
        catch
        {
            cores.Clear();
        }

        cores.RemoveAll(core => core.LogicalIds.Length == 0);
        cores.Sort(CompareCores);

        bool sawSmt = false;
        for (int i = 0; i < cores.Count; i++)
        {
            if (cores[i].LogicalIds.Length > 1)
            {
                sawSmt = true;
                break;
            }
        }

        SawSmt = sawSmt;
        PhysicalCoreCount = cores.Count;
        List<Core> pCores = SelectPCores(cores);
        int workers = ChooseWorkerCount(logical, cores.Count, pCores.Count, sawSmt, IsVirtualMachine);
        AutoHint = DescribeChoice(logical, cores.Count, sawSmt, IsVirtualMachine, workers);

        if (cores.Count == 0)
        {
            int[] all = new int[logical];
            for (int i = 0; i < logical; i++)
                all[i] = i;
            PCoreLogicalIds = all;
            PCoreLeaders = all;
            PhysicalLeaders = all;
            PhysicalLogicalIds = all;
            PreferPCoreCount = workers;
            return;
        }

        List<int> physicalLogical = [];
        List<int> physicalLeaders = [];
        foreach (Core core in cores)
        {
            physicalLeaders.Add(core.LogicalIds[0]);
            physicalLogical.AddRange(core.LogicalIds);
        }

        List<int> pLogical = [];
        List<int> pLeaders = [];
        foreach (Core core in pCores)
        {
            pLeaders.Add(core.LogicalIds[0]);
            pLogical.AddRange(core.LogicalIds);
        }

        PhysicalLeaders = physicalLeaders;
        PhysicalLogicalIds = physicalLogical;
        PCoreLogicalIds = pLogical;
        PCoreLeaders = pLeaders;
        PreferPCoreCount = workers;
    }

    /// <summary>
    /// Picks a worker count at or below the physical P-core count.
    /// A VM that hides SMT (every vCPU looks like its own core) is halved,
    /// because one extra worker per physical core collapses GEMM throughput.
    /// </summary>
    public static int ChooseWorkerCount(int logicalCount, int physicalCount, int pCoreCount, bool sawSmt, bool virtualMachine)
    {
        if (logicalCount <= 0)
            logicalCount = 1;

        if (physicalCount <= 0)
        {
            if (virtualMachine && logicalCount >= 2 && (logicalCount & 1) == 0)
                return logicalCount / 2;
            return logicalCount <= 4 ? logicalCount : logicalCount / 2;
        }

        int preferred = pCoreCount > 0 ? pCoreCount : physicalCount;
        if (virtualMachine && !sawSmt && physicalCount == logicalCount && logicalCount >= 2 && (logicalCount & 1) == 0)
            preferred = Math.Min(preferred, logicalCount / 2);
        return Math.Max(1, Math.Min(preferred, logicalCount));
    }

    /// <summary>
    /// Intel hybrid parts expose SMT on P-cores only. When that mix is
    /// visible, those cores are the P-set. Otherwise the lowest efficiency
    /// class is treated as the performance class (Linux <c>cpu_core</c> is 0).
    /// </summary>
    public static int CountPCores(int[] logicalsPerCore, byte[]? efficiencyClasses = null)
    {
        if (logicalsPerCore.Length == 0)
            return 0;

        bool anySmt = false;
        bool anySingle = false;
        for (int i = 0; i < logicalsPerCore.Length; i++)
        {
            if (logicalsPerCore[i] > 1)
                anySmt = true;
            else
                anySingle = true;
        }

        if (anySmt && anySingle)
        {
            int smt = 0;
            for (int i = 0; i < logicalsPerCore.Length; i++)
            {
                if (logicalsPerCore[i] > 1)
                    smt++;
            }

            return smt;
        }

        if (efficiencyClasses is null || efficiencyClasses.Length != logicalsPerCore.Length)
            return logicalsPerCore.Length;

        byte best = efficiencyClasses[0];
        for (int i = 1; i < efficiencyClasses.Length; i++)
        {
            if (efficiencyClasses[i] < best)
                best = efficiencyClasses[i];
        }

        int n = 0;
        for (int i = 0; i < efficiencyClasses.Length; i++)
        {
            if (efficiencyClasses[i] == best)
                n++;
        }

        return n;
    }

    public static int[] ParseCpuList(string text)
    {
        List<int> ids = [];
        foreach (string part in text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int dash = part.IndexOf('-');
            if (dash < 0)
            {
                if (int.TryParse(part, out int one))
                    ids.Add(one);
                continue;
            }

            if (!int.TryParse(part[..dash], out int begin) || !int.TryParse(part[(dash + 1)..], out int end))
                continue;
            if (end < begin)
                (begin, end) = (end, begin);
            for (int i = begin; i <= end; i++)
                ids.Add(i);
        }

        return [.. ids];
    }

    private static string DescribeChoice(int logical, int physical, bool sawSmt, bool virtualMachine, int workers)
    {
        if (physical <= 0)
            return virtualMachine ? "logical/2 (VM, no topology)" : "logical estimate";
        if (virtualMachine && !sawSmt && physical == logical && workers < physical)
            return "physical P-cores; VM without visible SMT";
        return "physical P-cores";
    }

    private static List<Core> SelectPCores(List<Core> cores)
    {
        if (cores.Count == 0)
            return cores;

        int[] logicals = new int[cores.Count];
        byte[] efficiency = new byte[cores.Count];
        for (int i = 0; i < cores.Count; i++)
        {
            logicals[i] = cores[i].LogicalIds.Length;
            efficiency[i] = cores[i].EfficiencyClass;
        }

        bool anySmt = false;
        bool anySingle = false;
        for (int i = 0; i < logicals.Length; i++)
        {
            if (logicals[i] > 1)
                anySmt = true;
            else
                anySingle = true;
        }

        List<Core> selected = [];
        if (anySmt && anySingle)
        {
            foreach (Core core in cores)
            {
                if (core.LogicalIds.Length > 1)
                    selected.Add(core);
            }

            return selected;
        }

        byte best = efficiency[0];
        for (int i = 1; i < efficiency.Length; i++)
        {
            if (efficiency[i] < best)
                best = efficiency[i];
        }

        foreach (Core core in cores)
        {
            if (core.EfficiencyClass == best)
                selected.Add(core);
        }

        return selected;
    }

    private static int CompareCores(Core a, Core b)
    {
        // SMT cores are P-cores on Intel hybrid parts; pin those first.
        int cmp = b.LogicalIds.Length.CompareTo(a.LogicalIds.Length);
        if (cmp != 0)
            return cmp;
        cmp = a.EfficiencyClass.CompareTo(b.EfficiencyClass);
        if (cmp != 0)
            return cmp;
        return a.LogicalIds[0].CompareTo(b.LogicalIds[0]);
    }

    private static void TryQueryWindows(List<Core> cores)
    {
        const int relationProcessorCore = 0;
        uint length = 0;
        GetLogicalProcessorInformationEx(relationProcessorCore, IntPtr.Zero, ref length);
        if (length == 0)
            return;

        IntPtr buffer = Marshal.AllocHGlobal((int)length);
        try
        {
            if (!GetLogicalProcessorInformationEx(relationProcessorCore, buffer, ref length))
                return;

            int offset = 0;
            while (offset + 8 <= (int)length)
            {
                int relationship = Marshal.ReadInt32(buffer, offset);
                int size = Marshal.ReadInt32(buffer, offset + 4);
                if (size < 32 || offset + size > (int)length)
                    break;
                if (relationship == relationProcessorCore)
                {
                    byte efficiency = Marshal.ReadByte(buffer, offset + 9);
                    ushort groupCount = (ushort)Marshal.ReadInt16(buffer, offset + 30);
                    List<int> ids = [];
                    int maskOffset = offset + 32;
                    for (int g = 0; g < groupCount && maskOffset + IntPtr.Size + 2 <= offset + size; g++)
                    {
                        nuint mask = unchecked((nuint)(nint)Marshal.ReadIntPtr(buffer, maskOffset));
                        ushort group = (ushort)Marshal.ReadInt16(buffer, maskOffset + IntPtr.Size);
                        for (int bit = 0; bit < 64; bit++)
                        {
                            if (((mask >> bit) & 1) != 0)
                                ids.Add(group * 64 + bit);
                        }

                        maskOffset += IntPtr.Size == 8 ? 16 : 12;
                    }

                    if (ids.Count > 0)
                        cores.Add(new Core(efficiency, [.. ids]));
                }

                offset += size;
            }
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    private static void TryQueryLinux(List<Core> cores)
    {
        int[] present = ReadLinuxPresentCpus();
        if (present.Length == 0)
            return;

        HashSet<int> pCoreLogicals = ReadLinuxPCoreLogicals();
        HashSet<int> seen = [];
        foreach (int id in present)
        {
            if (!seen.Add(id))
                continue;

            int[] siblings = ReadLinuxSiblings(id);
            if (siblings.Length == 0)
                siblings = [id];
            foreach (int sibling in siblings)
                seen.Add(sibling);

            byte efficiency = 0;
            if (pCoreLogicals.Count > 0)
            {
                efficiency = 1;
                foreach (int sibling in siblings)
                {
                    if (pCoreLogicals.Contains(sibling))
                    {
                        efficiency = 0;
                        break;
                    }
                }
            }

            cores.Add(new Core(efficiency, siblings));
        }
    }

    private static int[] ReadLinuxPresentCpus()
    {
        string[] paths =
        [
            "/sys/devices/system/cpu/present",
            "/sys/devices/system/cpu/online",
        ];
        foreach (string path in paths)
        {
            if (!File.Exists(path))
                continue;
            try
            {
                int[] ids = ParseCpuList(File.ReadAllText(path));
                if (ids.Length > 0)
                    return ids;
            }
            catch
            {
                // Best-effort.
            }
        }

        return [];
    }

    private static HashSet<int> ReadLinuxPCoreLogicals()
    {
        HashSet<int> ids = [];
        string[] paths =
        [
            "/sys/devices/cpu_core/cpus",
            "/sys/devices/system/cpu/cpu_core/cpus",
        ];
        foreach (string path in paths)
        {
            if (!File.Exists(path))
                continue;
            try
            {
                foreach (int id in ParseCpuList(File.ReadAllText(path)))
                    ids.Add(id);
                if (ids.Count > 0)
                    return ids;
            }
            catch
            {
                // Best-effort.
            }
        }

        return ids;
    }

    private static int[] ReadLinuxSiblings(int cpu)
    {
        string listPath = $"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list";
        try
        {
            if (File.Exists(listPath))
            {
                int[] ids = ParseCpuList(File.ReadAllText(listPath));
                if (ids.Length > 0)
                    return ids;
            }
        }
        catch
        {
            // Fall through.
        }

        return [cpu];
    }

    private static bool DetectVirtualMachine()
    {
        try
        {
            if (OperatingSystem.IsLinux() && HasLinuxHypervisorFlag())
                return true;

            if (X86Base.IsSupported)
            {
                (int eax, int ebx, int ecx, int edx) = X86Base.CpuId(1, 0);
                if ((ecx & (1 << 31)) != 0)
                {
                    if (OperatingSystem.IsLinux())
                        return true;

                    (eax, ebx, ecx, edx) = X86Base.CpuId(unchecked((int)0x40000000), 0);
                    if (IsGuestOnlyHypervisorVendor(CpuidDwords(ebx, ecx, edx)))
                        return true;
                }
            }

            if (OperatingSystem.IsWindows())
                return IsWindowsVirtualGuest();
        }
        catch
        {
            return false;
        }

        return false;
    }

    private static bool HasLinuxHypervisorFlag()
    {
        try
        {
            if (File.Exists("/sys/hypervisor/type"))
                return true;
            if (File.Exists("/proc/cpuinfo") && File.ReadAllText("/proc/cpuinfo").Contains("hypervisor"))
                return true;
        }
        catch
        {
            return false;
        }

        return false;
    }

    [SupportedOSPlatform("windows")]
    private static bool IsWindowsVirtualGuest()
    {
        try
        {
            // The Guest key also exists on a Hyper-V *host*. Parameters is guest-only.
            using RegistryKey? parameters = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\Microsoft\Virtual Machine\Guest\Parameters");
            string? name = Convert.ToString(parameters?.GetValue("VirtualMachineName"));
            if (!string.IsNullOrWhiteSpace(name))
                return true;
        }
        catch
        {
            // Best-effort.
        }

        try
        {
            using RegistryKey? bios = Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\BIOS");
            if (bios is not null)
            {
                string product = Convert.ToString(bios.GetValue("SystemProductName")) ?? "";
                string manufacturer = Convert.ToString(bios.GetValue("SystemManufacturer")) ?? "";
                if (LooksLikeVirtualFirmware(product, manufacturer))
                    return true;
            }
        }
        catch
        {
            return false;
        }

        return false;
    }

    public static bool LooksLikeVirtualFirmware(string product, string manufacturer)
    {
        string text = (product + " " + manufacturer).ToLowerInvariant();
        string[] needles =
        [
            "virtual machine", "virtualbox", "vmware", "kvm", "qemu", "xen",
            "hyper-v", "bochs", "parallels", "openstack", "google compute",
            "compute engine", "hvm domu", "alibaba", "qcloud", "tencent",
            "amazon ec2", "droplet",
        ];
        foreach (string needle in needles)
        {
            if (text.Contains(needle, StringComparison.Ordinal))
                return true;
        }

        return false;
    }

    private static bool IsGuestOnlyHypervisorVendor(string vendor)
    {
        string text = vendor.ToLowerInvariant();
        return text.Contains("vmware")
            || text.Contains("kvm")
            || text.Contains("xen")
            || text.Contains("vbox")
            || text.Contains("prl hyperv")
            || text.Contains("bhyve")
            || text.Contains("acrn");
    }

    private static string CpuidDwords(int ebx, int ecx, int edx)
    {
        Span<byte> bytes = stackalloc byte[12];
        BitConverter.TryWriteBytes(bytes[..4], ebx);
        BitConverter.TryWriteBytes(bytes[4..8], ecx);
        BitConverter.TryWriteBytes(bytes[8..12], edx);
        return Encoding.ASCII.GetString(bytes);
    }

    private readonly record struct Core(byte EfficiencyClass, int[] LogicalIds);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetLogicalProcessorInformationEx(
        int relationshipType,
        IntPtr buffer,
        ref uint returnedLength);
}
