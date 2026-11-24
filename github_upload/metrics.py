import numpy as np

if __package__:
    from . import config as C
else:
    import config as C

EVAL_START = C.WARMUP_DAYS * C.SLOTS


def confusion(flag, label):
    f = flag[:, EVAL_START:]
    y = label[:, EVAL_START:]
    tp = int(np.sum(f & y)); fn = int(np.sum(~f & y))
    fp = int(np.sum(f & ~y)); tn = int(np.sum(~f & ~y))
    return tp, fp, tn, fn


def evaluate(flag, label, events, benign=None):
    tp, fp, tn, fn = confusion(flag, label)
    dr = 100.0 * tp / max(tp + fn, 1)
    fpr = 100.0 * fp / max(fp + tn, 1)
    delays, detected = [], 0
    for e in events:
        m, a, b = e["meter"], e["start"], e["end"]
        hits = np.flatnonzero(flag[m, a:b] & label[m, a:b])
        if label[m, a:b].any():
            if hits.size:
                detected += 1
                first_attacked = np.flatnonzero(label[m, a:b])[0]
                delays.append(int(hits[0] - first_attacked))
    n_ev = sum(1 for e in events if label[e["meter"], e["start"]:e["end"]].any())
    out = dict(TP=tp, FP=fp, TN=tn, FN=fn, DR=dr, FPR=fpr,
               event_DR=100.0 * detected / max(n_ev, 1),
               delay=float(np.mean(delays)) if delays else np.nan,
               delay_median=float(np.median(delays)) if delays else np.nan,
               n_events=n_ev)
    if benign is not None:   
        fpm = flag[:, EVAL_START:] & ~label[:, EVAL_START:]
        b = benign[:, EVAL_START:]
        names = {0: "none", 1: "vacation", 2: "weather", 3: "legit_off", 4: "comms"}
        for code, name in names.items():
            out[f"FP_{name}"] = int(np.sum(fpm & (b == code)))
    return out
