import glob
import json
import statistics as st
import sys

for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + "/probe2/probe2-*.json")):
        r = json.load(open(f))

        def m(k, skip=0):
            v = r.get(k, [])[skip:]
            return round(st.median(v) * 1000, 2) if v else None

        nf = len(r["dit_call_s"])
        ns = len(r["step_s"])
        per = nf // ns if ns else 0
        print(f, "rank", r["rank"], "steps", ns, "fwd/step", per)
        print("  step_ms med(skip1)", m("step_s", 1), "all", [round(x * 1000, 1) for x in r["step_s"]])
        print("  dit_call_ms med(skip first step)", m("dit_call_s", per), " sched_ms", m("sched_s", 1),
              "all calls", [round(x * 1000, 1) for x in r["dit_call_s"]])
        if r["fwd_s"]:
            print("  h2d", m("h2d_s", per), "fwd", m("fwd_s", per), "d2h", m("d2h_s", per))
        if r["seg"]:
            keys = r["seg"][-1].keys()
            print("  seg", {k: round(st.median([s[k] for s in r["seg"][per:] if k in s]) * 1000, 2) for k in keys})
        ss = r["step_s"][1:]
        if ss:
            stepm = st.median(ss)
            callm = st.median(r["dit_call_s"][per:])
            sm = st.median(r["sched_s"][1:])
            print("  residual (step - fwd*n - sched) ms", round((stepm - per * callm - sm) * 1000, 2))
