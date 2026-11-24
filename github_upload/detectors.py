
import numpy as np

if __package__:
    from . import config as C
else:
    import config as C



def run_length(E):
    R = np.zeros(E.shape, dtype=np.int32)
    R[:, 0] = E[:, 0]
    for t in range(1, E.shape[1]):
        R[:, t] = (R[:, t - 1] + 1) * E[:, t]
    return R


def rolling_baseline(X, window_days=C.D1_WINDOW_DAYS):
    
    n = X.shape[0]
    Xd = X.reshape(n, C.N_DAYS, C.SLOTS).astype(np.float64)
    present = ~np.isnan(Xd)
    v = np.where(present, Xd, 0.0)
    cs1 = np.concatenate([np.zeros((n, 1, C.SLOTS)), np.cumsum(v, axis=1)], axis=1)
    cs2 = np.concatenate([np.zeros((n, 1, C.SLOTS)), np.cumsum(v * v, axis=1)], axis=1)
    csn = np.concatenate([np.zeros((n, 1, C.SLOTS)), np.cumsum(present, axis=1)], axis=1)
    d = np.arange(C.N_DAYS)
    lo = np.maximum(0, d - window_days)
    s1 = cs1[:, d, :] - cs1[:, lo, :]
    s2 = cs2[:, d, :] - cs2[:, lo, :]
    cnt = csn[:, d, :] - csn[:, lo, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / cnt
        var = s2 / cnt - mu ** 2
    mu[cnt < 7] = np.nan                       
    sd = np.sqrt(np.maximum(var, 0.0))
    sd = np.maximum(sd, 0.05 * mu + 0.01)      
    return mu.reshape(n, -1).astype(np.float32), sd.reshape(n, -1).astype(np.float32)


def _burst_state(status, min_count, group_size=None):

    n, T = status.shape
    off_tr = np.zeros((n, T), dtype=bool)
    off_tr[:, 1:] = (status[:, :-1] == 1) & (status[:, 1:] == 0)
    win = off_tr.copy()
    win[:, 1:] |= off_tr[:, :-1]               
    if group_size is None:
        burst_t = win.sum(axis=0) >= min_count              
        burst = win & burst_t[None, :]
    else:
        g = win.reshape(n // group_size, group_size, T)
        burst_g = g.sum(axis=1) >= min_count               
        burst = (g & burst_g[:, None, :]).reshape(n, T)
    active = np.zeros((n, T), dtype=bool)
    state = np.zeros(n, dtype=bool)
    for t in range(T):
        state = (state | burst[:, t]) & (status[:, t] == 0)
        active[:, t] = state
    return active


def d1_scores(X):
    mu, sd = rolling_baseline(X)
    with np.errstate(invalid="ignore"):
        z = (X - mu) / sd
    return z, mu


def d1_flags(z, tau=C.D1_Z, k=C.D1_K):
    with np.errstate(invalid="ignore"):
        E = np.abs(z) > tau                    
    return run_length(E) >= k


def d2_rules(X, status):
    n, T = X.shape
    present = ~np.isnan(X)
    rules = {}
    rules["R-GAP"] = run_length(~present) >= C.D2_GAP

    same = np.zeros((n, T), dtype=bool)
    same[:, 1:] = (X[:, 1:] == X[:, :-1]) & (status[:, 1:] == 1)
    rules["R-REPEAT"] = run_length(same) >= C.D2_REPEAT - 1

    rep = np.zeros((n, T), dtype=bool)
    for lag in (C.SLOTS, 7 * C.SLOTS):
        eq = np.zeros((n, T), dtype=bool)
        eq[:, lag:] = (X[:, lag:] == X[:, :-lag]) & (X[:, lag:] > 0)
        both = np.zeros((n, T), dtype=bool)
        both[:, 1:] = eq[:, 1:] & eq[:, :-1]
        rep |= both
    rules["R-REPLAY"] = rep

    with np.errstate(invalid="ignore"):
        rules["R-RANGE"] = (X < 0) | (X > C.D2_MAX_KWH)
    rules["R-BURST"] = _burst_state(status, int(np.floor(C.D2_BURST_FRACTION * n)) + 1)
    return rules


def d3_value_scores(X, mu):
    n, T = X.shape
    G = n // C.GROUP_SIZE
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (X / mu).reshape(G, C.GROUP_SIZE, T)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(r, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(r - med), axis=1, keepdims=True)
    s = np.maximum(1.4826 * mad, 0.05)
    return ((r - med) / s).reshape(n, T)


def d3_rules(X, status, score, tau=C.D3_Z, k=C.D3_K):
    n, T = X.shape
    G = n // C.GROUP_SIZE
    rules = {}
    with np.errstate(invalid="ignore"):
        rules["N-VALUE"] = run_length(np.abs(score) > tau) >= k
    miss = np.isnan(X).reshape(G, C.GROUP_SIZE, T)
    co = miss.sum(axis=1, keepdims=True) >= C.D3_MIN_SILENT
    rules["N-SILENT"] = (miss & co).reshape(n, T)
    rules["N-STATUS"] = _burst_state(status, C.D3_STATUS_MIN, C.GROUP_SIZE)
    return rules


def run_all(X, status):
    z, mu = d1_scores(X)
    f1 = d1_flags(z)
    r2 = d2_rules(X, status)
    s3 = d3_value_scores(X, mu)
    r3 = d3_rules(X, status, s3)
    f2 = np.logical_or.reduce(list(r2.values()))
    f3 = np.logical_or.reduce(list(r3.values()))
    flags = {"D1": f1, "D2": f2, "D3": f3, "Combined": f1 | f2 | f3}
    return flags, {**r2, **r3}, {"z": z, "mu": mu, "s3": s3}


def cusum_low_flags(z, kappa=1.0, h=10.0, clip=6.0, cap=1.5):
    
    zc = np.clip(np.nan_to_num(-z, nan=kappa), -clip, clip)   
    S = np.zeros(z.shape[0]); out = np.zeros(z.shape, dtype=bool)
    for t in range(z.shape[1]):
        S = np.minimum(np.maximum(0.0, S + zc[:, t] - kappa), cap * h)
        out[:, t] = S > h
    return out
