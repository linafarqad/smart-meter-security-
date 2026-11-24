
import sys, time, os
import numpy as np
import pandas as pd
from multiprocessing import Pool


if __package__:
    from . import config as C, generator, attacks, detectors as D, metrics
else:
    import config as C, generator, attacks, detectors as D, metrics as metrics

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
SCENARIOS = ["S1", "S2", "S3", "S4"]
E1_KAPPA, E1_H = 1.0, 20.0          

def detect(X, status):
    
    tm = {}
    t = time.perf_counter(); z, mu = D.d1_scores(X); f1 = D.d1_flags(z); tm["D1"] = time.perf_counter() - t
    t = time.perf_counter(); r2 = D.d2_rules(X, status); tm["D2"] = time.perf_counter() - t
    t = time.perf_counter(); s3 = D.d3_value_scores(X, mu); r3 = D.d3_rules(X, status, s3)
    tm["D3"] = time.perf_counter() - t + tm["D1"] * 0.5      # D3 reuses the D1 baseline
    t = time.perf_counter(); e1 = D.cusum_low_flags(s3, E1_KAPPA, E1_H); tm["E1"] = time.perf_counter() - t
    f2 = np.logical_or.reduce(list(r2.values())); f3 = np.logical_or.reduce(list(r3.values()))
    flags = {"D1": f1, "D2": f2, "D3": f3, "E1": e1,
             "Combined": f1 | f2 | f3, "Combined+E1": f1 | f2 | f3 | e1}
    return flags, {**r2, **r3}, tm, (z, mu, s3)


def variant_of(e):
    return e.get("mode") or e.get("variant") or "-"


def main_job(seed):
    rows, rule_rows, var_rows = [], [], []
    data = generator.generate(seed)
    for k, sc in enumerate(SCENARIOS):
        att = attacks.inject(data, sc, np.random.default_rng(10_000 * seed + k))
        flags, rules, tm, _ = detect(att["X"], att["status"])
        for name, f in flags.items():
            r = metrics.evaluate(f, att["label"], att["events"], data["benign"])
            r.update(seed=seed, scenario=sc, method=name, runtime_s=tm.get(name, np.nan))
            rows.append(r)
            for v in sorted({variant_of(e) for e in att["events"]}):
                evs = [e for e in att["events"] if variant_of(e) == v]
                rv = metrics.evaluate(f, att["label"], evs)
                var_rows.append(dict(seed=seed, scenario=sc, method=name, variant=v,
                                     event_DR=rv["event_DR"], delay=rv["delay"]))
        for name, f in rules.items():
            r = metrics.evaluate(f, att["label"], att["events"])
            rule_rows.append(dict(seed=seed, scenario=sc, rule=name, DR=r["DR"], FPR=r["FPR"]))
    return rows, rule_rows, var_rows
 

SWEEPS = {
    "S1": ("alpha", [0.5, 0.6, 0.7, 0.8, 0.9], {"fraction": 0.10, "days": 3}),
    "S2": ("hours", [2, 4, 8, 16, 24], {}),
    "S3": ("hours", [6, 12, 24, 36, 48], {"fraction": 0.10}),
    "S4": ("fraction", [0.05, 0.10, 0.15, 0.20, 0.30], {}),
}


def sweep_job(args):
    seed, sc, value = args
    pname, _, base = SWEEPS[sc]
    data = generator.generate(seed)
    fixed = dict(base); fixed[pname] = value
    att = attacks.inject(data, sc, np.random.default_rng(777 + seed), fixed=fixed)
    flags, _, _, _ = detect(att["X"], att["status"])
    out = []
    for name in ["D1", "D2", "D3", "E1", "Combined", "Combined+E1"]:
        r = metrics.evaluate(flags[name], att["label"], att["events"])
        out.append(dict(seed=seed, scenario=sc, param=pname, value=value, method=name,
                        DR=r["DR"], FPR=r["FPR"], event_DR=r["event_DR"], delay=r["delay"]))
    return out


def threshold_job(seed):
    data = generator.generate(seed)
    att = attacks.inject(data, "S1", np.random.default_rng(555 + seed))
    z, mu = D.d1_scores(att["X"]); s3 = D.d3_value_scores(att["X"], mu)
    out = []
    for tau in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        r = metrics.evaluate(D.d1_flags(z, tau, C.D1_K), att["label"], att["events"])
        out.append(dict(seed=seed, method="D1", thr=tau, DR=r["DR"], FPR=r["FPR"], delay=r["delay"]))
    for tau in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        with np.errstate(invalid="ignore"):
            f = D.run_length(np.abs(s3) > tau) >= C.D3_K
        r = metrics.evaluate(f, att["label"], att["events"])
        out.append(dict(seed=seed, method="D3 (N-VALUE)", thr=tau, DR=r["DR"], FPR=r["FPR"], delay=r["delay"]))
    for h in [5, 10, 20, 40, 80]:
        r = metrics.evaluate(D.cusum_low_flags(s3, E1_KAPPA, h), att["label"], att["events"])
        out.append(dict(seed=seed, method="E1 (CUSUM)", thr=h, DR=r["DR"], FPR=r["FPR"], delay=r["delay"]))
    for k in [1, 2, 3, 4, 6]:
        r = metrics.evaluate(D.d1_flags(z, C.D1_Z, k), att["label"], att["events"])
        out.append(dict(seed=seed, method="D1 k (tau=3)", thr=k, DR=r["DR"], FPR=r["FPR"], delay=r["delay"]))
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    with Pool(2) as pool:
        if what in ("main", "all"):
            res = pool.map(main_job, range(C.N_SEEDS))
            pd.DataFrame([r for a, _, _ in res for r in a]).to_csv(f"{OUT}/main.csv", index=False)
            pd.DataFrame([r for _, b, _ in res for r in b]).to_csv(f"{OUT}/main_rules.csv", index=False)
            pd.DataFrame([r for _, _, c in res for r in c]).to_csv(f"{OUT}/main_variants.csv", index=False)
            print("main done", round(time.time() - t0), "s", flush=True)
        if what in ("sweep", "all"):
            jobs = [(s, sc, v) for sc, (_, vals, _) in SWEEPS.items() for v in vals for s in range(10)]
            res = pool.map(sweep_job, jobs)
            pd.DataFrame([r for a in res for r in a]).to_csv(f"{OUT}/sweep.csv", index=False)
            print("sweep done", round(time.time() - t0), "s", flush=True)
        if what in ("threshold", "all"):
            res = pool.map(threshold_job, range(10))
            pd.DataFrame([r for a in res for r in a]).to_csv(f"{OUT}/threshold.csv", index=False)
            print("threshold done", round(time.time() - t0), "s", flush=True)

