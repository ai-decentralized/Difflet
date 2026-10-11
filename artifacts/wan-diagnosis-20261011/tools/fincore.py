#!/usr/bin/env python3
"""Page-cache residency of files (mincore), like util-linux fincore."""
import ctypes, os, sys
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
PAGE = os.sysconf("SC_PAGE_SIZE"); PROT_READ = 1; MAP_SHARED = 1
tot_r = tot_s = 0
for path in sys.argv[1:]:
    size = os.path.getsize(path)
    if size == 0:
        continue
    fd = os.open(path, os.O_RDONLY)
    addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
    npages = (size + PAGE - 1) // PAGE
    vec = (ctypes.c_ubyte * npages)()
    if libc.mincore(addr, size, vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore")
    res = sum(bytes(vec)) if False else bytes(vec).count(1) + sum(1 for b in bytes(vec) if b > 1)
    libc.munmap(addr, size); os.close(fd)
    tot_r += res * PAGE; tot_s += size
    print(f"{res*PAGE/2**30:8.2f} / {size/2**30:8.2f} GiB  {100*res/npages:5.1f}%  {os.path.basename(os.path.dirname(path))[:60]}/{os.path.basename(path)}")
if len(sys.argv) > 2:
    print(f"{tot_r/2**30:8.2f} / {tot_s/2**30:8.2f} GiB total")
