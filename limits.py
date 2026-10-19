"""Pi-like limits for testing on the PC: use only N CPU cores, and report the peak memory the app needed.
This only approximates a Raspberry Pi 5 (4 cores, 8 GB): the PC cores are faster and a different chip type, so timings stay a guide, not a proof."""
import ctypes
import os


def limit_cores(n):
    """Pin this process to n logical CPUs, spread apart so they are different physical cores on a hyper-threaded PC. Returns a short description."""
    if n <= 0:
        return "no core limit"
    total = os.cpu_count() or n
    n = min(n, total)
    step = max(1, total // n)
    cpus = [i * step for i in range(n)]
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cpus))
        else:
            mask = 0
            for c in cpus:
                mask |= 1 << c
            k = ctypes.windll.kernel32
            k.GetCurrentProcess.restype = ctypes.c_void_p
            k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            if not k.SetProcessAffinityMask(k.GetCurrentProcess(), mask):
                return "core limit failed"
        return f"limited to {n} of {total} logical CPUs ({cpus})"
    except Exception as ex:
        return f"core limit failed: {ex!r}"


def peak_memory_mb():
    """Peak working set of this process in MB (Windows), or None."""
    try:
        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        p = ctypes.windll.psapi
        p.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        if p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return round(pmc.PeakWorkingSetSize / 1048576, 0)
    except Exception:
        pass
    return None


def cpu_seconds():
    """CPU time used by this process so far (all threads)."""
    import time
    return time.process_time()
