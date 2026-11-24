
import numpy as np

if __package__:
    from . import config as C
else:
    import config as C

def _campaign_starts(rng, n_campaigns, max_len):
    
    first = C.WARMUP_DAYS * C.SLOTS
    window = (C.T - first) // n_campaigns
    starts = []
    for k in range(n_campaigns):
        lo = first + k * window
        hi = lo + window - max_len - C.SLOTS
        starts.append(int(rng.integers(lo, max(hi, lo + 1))))
    return starts


def _u(rng, rng_pair, fixed=None):
    return fixed if fixed is not None else rng.uniform(*rng_pair)


def inject(data, scenario, rng, fixed=None):

    fixed = fixed or {}
    X = data["X"].copy()
    status = data["status"].copy()
    label = np.zeros(X.shape, dtype=bool)
    events = []                        
    n = C.N_METERS
    starts = _campaign_starts(rng, C.CAMPAIGNS_PER_RUN, max_len=7 * C.SLOTS)

    for c, t0 in enumerate(starts):
        if scenario == "S1":
            frac = _u(rng, C.S1_FRACTION, fixed.get("fraction"))
            alpha = _u(rng, C.S1_ALPHA, fixed.get("alpha"))
            beta = rng.uniform(*C.S1_BETA)
            L = int(round(_u(rng, C.S1_DAYS, fixed.get("days")) * C.SLOTS))
            meters = rng.choice(n, max(1, int(round(frac * n))), replace=False)
            for m in meters:
                seg = X[m, t0:t0 + L]
                present = ~np.isnan(seg)
                X[m, t0:t0 + L] = np.where(present, np.maximum(0.0, alpha * seg + beta), np.nan)
                label[m, t0:t0 + L] = present
                events.append(dict(meter=int(m), start=t0, end=t0 + L, campaign=c,
                                   alpha=alpha, frac=frac))

        elif scenario == "S2":
            L = int(round(_u(rng, C.S2_HOURS, fixed.get("hours")) * 2))
            mode = fixed.get("mode") or ("single" if rng.random() < 0.5 else "group")
            if mode == "single":
                k = int(rng.integers(C.S2_SINGLE_METERS[0], C.S2_SINGLE_METERS[1] + 1))
                meters = rng.choice(n, k, replace=False)
            else:
                groups = rng.choice(n // C.GROUP_SIZE,
                                    int(rng.integers(C.S2_GROUPS[0], C.S2_GROUPS[1] + 1)),
                                    replace=False)
                frac = rng.uniform(*C.S2_GROUP_FRACTION)
                k = max(2, int(round(frac * C.GROUP_SIZE)))
                meters = np.concatenate([g * C.GROUP_SIZE + rng.choice(C.GROUP_SIZE, k, replace=False)
                                         for g in groups])
            for m in meters:
                present = ~np.isnan(X[m, t0:t0 + L])
                label[m, t0:t0 + L] = present
                X[m, t0:t0 + L] = np.nan
                events.append(dict(meter=int(m), start=t0, end=t0 + L, campaign=c, mode=mode))

        elif scenario == "S3":
            frac = _u(rng, C.S3_FRACTION, fixed.get("fraction"))
            L = int(round(_u(rng, C.S3_HOURS, fixed.get("hours")) * 2))
            variant = fixed.get("variant") or ("freeze" if rng.random() < 0.5 else "replay")
            meters = rng.choice(n, max(1, int(round(frac * n))), replace=False)
            for m in meters:
                a, b = t0 + 1, t0 + 1 + L         
                if variant == "freeze":
                    hist = X[m, :a]
                    last = hist[~np.isnan(hist)][-1]
                    new = np.full(L, last, dtype=np.float32)
                else:
                    new = X[m, a - C.S3_REPLAY_LAG:b - C.S3_REPLAY_LAG].copy()
                present = ~np.isnan(X[m, a:b]) & ~np.isnan(new)
                X[m, a:b] = np.where(present, new, np.nan)
                label[m, a:b] = present
                events.append(dict(meter=int(m), start=a, end=b, campaign=c, variant=variant))

        elif scenario == "S4":
            frac = _u(rng, C.S4_FRACTION, fixed.get("fraction"))
            off = int(rng.integers(C.S4_OFF_INTERVALS[0], C.S4_OFF_INTERVALS[1] + 1))
            meters = rng.choice(n, max(1, int(round(frac * n))), replace=False)
            for m in meters:
                if status[m, t0 - 1] == 0:          
                    continue
                a = t0 + int(rng.integers(0, 2))    
                b = a + off
                status[m, a:b] = 0
                present = ~np.isnan(X[m, a:b])
                X[m, a:b] = np.where(present, 0.0, np.nan)
                label[m, a:b] = present
                events.append(dict(meter=int(m), start=a, end=b, campaign=c, frac=frac))
        else:
            raise ValueError(scenario)

    return {"X": X, "status": status, "label": label, "events": events}
