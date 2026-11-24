
import numpy as np

if __package__:
    from . import config as C
else:
    import config as C


def daily_profile(rng, n):

    h = np.arange(C.SLOTS) / 2.0                      # hour of day 0.0 ... 23.5
    morning_amp = rng.uniform(0.6, 1.2, (n, 1))
    evening_amp = rng.uniform(1.0, 1.8, (n, 1))
    shift = rng.normal(0.0, 0.5, (n, 1))              # +/- 30 min habits
    base = 0.45
    morning = morning_amp * np.exp(-0.5 * ((h - 8.0 - shift) / 1.0) ** 2)
    evening = evening_amp * np.exp(-0.5 * ((h - 19.0 - shift) / 1.6) ** 2)
    daytime = 0.25 * np.exp(-0.5 * ((h - 13.0) / 3.0) ** 2)
    prof = base + morning + evening + daytime
    return prof / prof.mean(axis=1, keepdims=True)


def calendar_factors():
    
    day = np.arange(C.N_DAYS)
    weekly = np.where(day % 7 >= 5, C.WEEKEND_FACTOR, 1.0)
    seasonal = 1.0 + C.SEASONAL_AMPLITUDE * np.cos(2 * np.pi * (day - 15) / 365.0)
    return np.repeat(weekly, C.SLOTS), np.repeat(seasonal, C.SLOTS)


def _random_windows(rng, rate_per_day, length_range, unit):
    
    n = rng.poisson(rate_per_day * C.N_DAYS)
    out = []
    for _ in range(n):
        length = int(rng.integers(length_range[0], length_range[1] + 1)) * unit
        start = int(rng.integers(0, C.T - length))
        out.append((start, start + length))
    return out


def generate(seed):
    
    rng = np.random.default_rng(seed)
    n, T = C.N_METERS, C.T

    mu_log = np.log(C.MEAN_DAILY_KWH / C.SLOTS) - C.LOGNORM_SIGMA ** 2 / 2
    B = rng.lognormal(mu_log, C.LOGNORM_SIGMA, n)                 # kWh per interval
    D = daily_profile(rng, n)                                     # (n, 48)
    W, S = calendar_factors()                                     # (T,)

    expected = B[:, None] * np.tile(D, (1, C.N_DAYS)) * (W * S)[None, :]
    benign = np.zeros((n, T), dtype=np.int8)   # 0 none, 1 vacation, 2 weather, 3 legit-off, 4 comms

    
    for _ in range(C.WEATHER_EVENTS):
        days = int(rng.integers(C.WEATHER_DAYS[0], C.WEATHER_DAYS[1] + 1))
        start = int(rng.integers(0, C.N_DAYS - days)) * C.SLOTS
        f = rng.uniform(*C.WEATHER_FACTOR)
        expected[:, start:start + days * C.SLOTS] *= f
        benign[:, start:start + days * C.SLOTS] = 2

    
    for i in range(n):
        for (a, b) in _random_windows(rng, C.VACATION_PER_YEAR / 365, C.VACATION_DAYS, C.SLOTS):
            expected[i, a:b] *= rng.uniform(*C.VACATION_FACTOR)
            benign[i, a:b] = 1

    eps = rng.normal(0.0, C.NOISE_SD, (n, T))
    X = np.maximum(0.0, expected * (1.0 + eps))
    X = np.round(X / C.RESOLUTION_KWH) * C.RESOLUTION_KWH
    status = np.ones((n, T), dtype=np.int8)

    
    for i in range(n):
        for (a, b) in _random_windows(rng, C.LEGIT_DISCONNECT_PER_YEAR / 365,
                                      C.LEGIT_DISCONNECT_DAYS, C.SLOTS):
            status[i, a:b] = 0
            X[i, a:b] = 0.0
            benign[i, a:b] = 3

    X = X.astype(np.float32)
    X[rng.random((n, T)) < C.P_PACKET_LOSS] = np.nan
    for i in range(n):
        for (a, b) in _random_windows(rng, C.METER_OUTAGE_PER_DAY, C.METER_OUTAGE_LEN, 1):
            X[i, a:b] = np.nan
            benign[i, a:b] = 4
    for g in range(n // C.GROUP_SIZE):
        rows = slice(g * C.GROUP_SIZE, (g + 1) * C.GROUP_SIZE)
        for (a, b) in _random_windows(rng, C.GROUP_OUTAGE_PER_DAY, C.GROUP_OUTAGE_LEN, 1):
            X[rows, a:b] = np.nan
            benign[rows, a:b] = 4

    return {"X": X, "status": status, "benign": benign, "B": B, "rng": rng}
