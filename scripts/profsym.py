#!/usr/bin/env python3
# Summarizes a WARPSTORE_PROFILE file: the functions with the most samples, as the leaf (self) and
# anywhere on the stack (total). Usage: scripts/profsym.py <profile> [binary] [--top N] [--callers FUNC]
import collections, os, subprocess, sys

args, i = [], 1
while i < len(sys.argv):  # (positional arguments; --top / --callers take a value)
    if sys.argv[i].startswith("--"):
        i += 2
        continue
    args.append(sys.argv[i])
    i += 1
prof = args[0]
binary = args[1] if len(args) > 1 else os.path.join(os.path.dirname(__file__), "..", "build", "warpstore")
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 30
callers = sys.argv[sys.argv.index("--callers") + 1] if "--callers" in sys.argv else None

stacks = []
for line in open(prof):
    f = line.split()
    stacks.append((int(f[0]), f[1:]))
own = os.path.basename(binary)
offs = sorted({fr.split("+", 1)[1] for _, st in stacks for fr in st if fr.startswith(own + "+")})
names = {}
if offs:  # (no debug info needed: the symbol table, nearest symbol at or below)
    import bisect
    syms = []
    for line in subprocess.run(["nm", "-n", "-C", "--defined-only", binary], capture_output=True, text=True).stdout.splitlines():
        f = line.split(" ", 2)
        if len(f) == 3 and f[1] in "tTwW":
            syms.append((int(f[0], 16), f[2]))
    addrs = [a for a, _ in syms]
    for o in offs:
        i = bisect.bisect_right(addrs, int(o, 16)) - 1
        names[own + "+" + o] = syms[i][1][:140] if i >= 0 else "?"


libs = {}  # shared objects: their dynamic symbols (path from ldd)
for line in subprocess.run(["ldd", binary], capture_output=True, text=True).stdout.splitlines():
    f = line.split()
    if len(f) >= 3 and f[1] == "=>" and f[2].startswith("/"):
        libs[os.path.basename(f[2])] = f[2]
libsyms = {}


def libname(fr):
    lib, off = fr.split("+", 1)
    if lib not in libs:
        return lib
    if lib not in libsyms:
        import bisect
        ss = []
        for line in subprocess.run(["nm", "-D", "-n", "-C", "--defined-only", libs[lib]], capture_output=True, text=True).stdout.splitlines():
            f = line.split(" ", 2)
            if len(f) == 3 and f[1] in "tTwWiI":
                ss.append((int(f[0], 16), f[2]))
        libsyms[lib] = ([a for a, _ in ss], ss)
    import bisect
    addrs, ss = libsyms[lib]
    i = bisect.bisect_right(addrs, int(off, 16)) - 1
    return f"{lib}:{ss[i][1]}" if i >= 0 else lib


def nm(fr):
    if fr not in names and not fr.startswith(own + "+"):
        names[fr] = libname(fr)

    return names.get(fr, fr.split("+")[0])


total = sum(c for c, _ in stacks)
selfc, inc = collections.Counter(), collections.Counter()
for c, st in stacks:
    if not st:
        continue
    selfc[nm(st[0])] += c
    for n in set(nm(fr) for fr in st):
        inc[n] += c
print(f"{total} samples (~ms of CPU)")
print("-- self")
for n, c in selfc.most_common(top):
    print(f"{100 * c / total:5.1f}%  {n}")
print("-- total")
for n, c in inc.most_common(top):
    print(f"{100 * c / total:5.1f}%  {n}")
if callers:
    cc = collections.Counter()
    for c, st in stacks:
        names_ = [nm(fr) for fr in st]
        for i, n in enumerate(names_):
            if callers in n and i + 1 < len(names_):
                cc[names_[i + 1]] += c
                break
    print(f"-- callers of {callers}")
    for n, c in cc.most_common(top):
        print(f"{100 * c / total:5.1f}%  {n}")
