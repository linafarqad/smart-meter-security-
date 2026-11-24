
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

if __package__:
    from . import config as C, generator, attacks, detectors as D
else:
    import config as C, generator, attacks, detectors as D

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(FIG, exist_ok=True)

# validated categorical palette, fixed order (slot 1..8)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHODS = ["D1", "D2", "D3", "E1", "Combined", "Combined+E1"]
MCOL = dict(zip(METHODS, PAL))
LABEL = {"D1": "D1 Rolling z-score", "D2": "D2 Rule-based", "D3": "D3 Neighbourhood",
         "E1": "E1 CUSUM (extension)", "Combined": "Combined D1+D2+D3",
         "Combined+E1": "Combined + E1"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 220,
    "savefig.bbox": "tight", "lines.linewidth": 1.6,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), facecolor="white")
    plt.close(fig)
    print("saved", name)


def day_axis(ax, t0, t1, step_days=1):
    ticks = np.arange(t0, t1 + 1, step_days * C.SLOTS)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"Day {t // C.SLOTS}" for t in ticks], fontsize=8)


# =============================================================== illustrative
def illustrative():
    data = generator.generate(0)
    X, benign = data["X"], data["benign"]

    # ---- Fig 13: clean data, one week and a full year
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), gridspec_kw={"height_ratios": [1, 1]})
    a, b = 70 * C.SLOTS, 77 * C.SLOTS
    tt = np.arange(a, b)
    for k, m in enumerate([3, 11, 17]):
        axes[0].plot(tt, X[m, a:b], color=PAL[k], lw=1.1, label=f"Household {m}")
    axes[0].plot(tt, np.nanmean(X[0:20, a:b], axis=0), color=INK, lw=2.0, label="Group 1 mean (20 meters)")
    day_axis(axes[0], a, b)
    axes[0].set_ylabel("kWh per 30 min"); axes[0].set_title("(a) One week (days 70-76): daily peaks, weekend uplift and household diversity")
    axes[0].legend(ncol=2, fontsize=8, loc="upper left")
    daily = np.nansum(X.reshape(C.N_METERS, C.N_DAYS, C.SLOTS), axis=2)
    pop = daily.mean(axis=0)
    axes[1].plot(np.arange(C.N_DAYS), pop, color=PAL[0], lw=1.4, label="Population mean")
    hv = int(np.flatnonzero((benign == 1)[:, C.WARMUP_DAYS * C.SLOTS:].any(axis=1))[0])
    axes[1].plot(np.arange(C.N_DAYS), daily[hv], color=PAL[1], lw=0.9, alpha=0.9, label=f"Household {hv}")
    w = (benign[0].reshape(C.N_DAYS, C.SLOTS) == 2).any(axis=1)
    for d in np.flatnonzero(w):
        axes[1].axvspan(d, d + 1, color=PAL[3], alpha=0.18, lw=0)
    v = (benign[hv].reshape(C.N_DAYS, C.SLOTS) == 1).any(axis=1)
    for d in np.flatnonzero(v):
        axes[1].axvspan(d, d + 1, color=PAL[4], alpha=0.25, lw=0)
    axes[1].axvline(C.WARMUP_DAYS, color=INK2, ls="--", lw=1)
    axes[1].text(C.WARMUP_DAYS + 2, axes[1].get_ylim()[1] * 0.92, "end of warm-up", fontsize=8, color=INK2)
    axes[1].set_xlabel("Day of synthetic year"); axes[1].set_ylabel("kWh per day")
    axes[1].set_title("(b) Full year: winter maximum, weather events (yellow) and a vacation (pink)")
    axes[1].legend(handles=[plt.Line2D([], [], color=PAL[0], label="Population mean"),
                            plt.Line2D([], [], color=PAL[1], label=f"Household {hv}"),
                            Patch(color=PAL[3], alpha=0.35, label="Weather event"),
                            Patch(color=PAL[4], alpha=0.4, label=f"Vacation (household {hv})")],
                   ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout(); save(fig, "fig13_synthetic_data.png")

    # ---- Fig 14: actual injected attacks, one example per scenario
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    ex = {}
    for sc, fixed in [("S1", {"alpha": 0.6, "days": 2, "fraction": 0.05}),
                      ("S2", {"hours": 10, "mode": "group"}),
                      ("S3", {"hours": 24, "variant": "freeze", "fraction": 0.05}),
                      ("S4", {"fraction": 0.2})]:
        att = attacks.inject(data, sc, np.random.default_rng(3), fixed=fixed)
        e = att["events"][0]
        ex[sc] = (att, e)
    titles = {"S1": "S1 False data injection (alpha = 0.6)", "S2": "S2 Communication interruption (10 h)",
              "S3": "S3 Frozen data (24 h)", "S4": "S4 Simultaneous status change"}
    for ax, sc in zip(axes.ravel(), ["S1", "S2", "S3", "S4"]):
        att, e = ex[sc]
        m, s0, s1 = e["meter"], e["start"], e["end"]
        a, b = s0 - 36, s1 + 36
        tt = np.arange(a, b)
        ax.plot(tt, X[m, a:b], color=INK2, lw=1.0, ls="--", label="True consumption")
        ax.plot(tt, att["X"][m, a:b], color=PAL[0], lw=1.5, label="Reported to MDMS")
        ax.axvspan(s0, s1, color=PAL[7], alpha=0.10, lw=0, label="Attack window")
        if sc == "S2":
            miss = np.isnan(att["X"][m, a:b])
            ax.scatter(tt[miss], np.zeros(miss.sum()), s=6, color=PAL[7], label="Missing reading")
        if sc == "S4":
            ax2 = ax.twinx(); ax2.grid(False)
            ax2.step(tt, att["status"][m, a:b], color=PAL[1], where="post", lw=1.3)
            ax2.set_ylim(-0.1, 3.0); ax2.set_yticks([0, 1]); ax2.set_yticklabels(["off", "on"], fontsize=8, color=PAL[1])
            ax2.spines["right"].set_visible(True)
            ax.plot([], [], color=PAL[1], label="Status flag")
        ax.set_title(titles[sc], fontsize=10)
        if sc == "S4":
            ax.set_xticks([s0]); ax.set_xticklabels(["start"], fontsize=8)
        else:
            ax.set_xticks([s0, s1]); ax.set_xticklabels(["start", "end"], fontsize=8)
        ax.set_ylabel("kWh / 30 min", fontsize=8)
    hd = [plt.Line2D([], [], color=INK2, ls="--", label="True consumption"),
          plt.Line2D([], [], color=PAL[0], label="Reported to MDMS"),
          Patch(color=PAL[7], alpha=0.2, label="Attack window"),
          plt.Line2D([], [], color=PAL[7], marker="o", ls="", ms=3, label="Missing reading (S2)"),
          plt.Line2D([], [], color=PAL[1], label="Status flag (S4)")]
    fig.legend(handles=hd, ncol=5, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "fig14_attack_examples.png")

    # ---- Fig 15: attack calendar - 8 campaigns per scenario in one seed
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for k, sc in enumerate(["S1", "S2", "S3", "S4"]):
        att = attacks.inject(data, sc, np.random.default_rng(10_000 * 0 + k))
        camp = {}
        for e in att["events"]:
            c = camp.setdefault(e["campaign"], [e["start"], e["end"], 0])
            c[0] = min(c[0], e["start"]); c[1] = max(c[1], e["end"]); c[2] += 1
        for c, (s0, s1, cnt) in camp.items():
            ax.barh(k, max((s1 - s0) / C.SLOTS, 0.6), left=s0 / C.SLOTS, height=0.5, color=PAL[k])
            ax.text(s0 / C.SLOTS, k + 0.33, str(cnt), fontsize=7, color=INK2)
    ax.axvspan(0, C.WARMUP_DAYS, color=GRID, alpha=0.7, lw=0)
    ax.text(4, 3.55, "warm-up (no attacks)", fontsize=8, color=INK2)
    ax.set_yticks(range(4)); ax.set_yticklabels(["S1 FDI", "S2 Interruption", "S3 Replay/freeze", "S4 Status burst"])
    ax.set_xlim(0, 365); ax.set_ylim(-0.6, 3.9); ax.invert_yaxis(); ax.set_xlabel("Day of synthetic year")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, "fig15_attack_calendar.png")

    # ---- Fig 17: D1 in action, Fig 18: D3 in action, Fig 20: E1 CUSUM
    att = attacks.inject(data, "S1", np.random.default_rng(5), fixed={"alpha": 0.55, "days": 3, "fraction": 0.05})
    Xa = att["X"]; z, mu = D.d1_scores(Xa); f1 = D.d1_flags(z)
    _, sd = D.rolling_baseline(Xa)
    e = att["events"][0]; m = e["meter"]; s0, s1 = e["start"], e["end"]
    a, b = s0 - 2 * C.SLOTS, s1 + 2 * C.SLOTS; tt = np.arange(a, b)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    axes[0].fill_between(tt, mu[m, a:b] - 3 * sd[m, a:b], mu[m, a:b] + 3 * sd[m, a:b], color=PAL[0], alpha=0.12, lw=0, label="Baseline mean ± 3σ")
    axes[0].plot(tt, mu[m, a:b], color=PAL[0], lw=1.0, ls="--", label="Baseline mean (28 days, same slot)")
    axes[0].plot(tt, Xa[m, a:b], color=INK, lw=1.1, label="Reported reading")
    fl = f1[m, a:b]
    axes[0].scatter(tt[fl], Xa[m, a:b][fl], s=10, color=PAL[7], zorder=5, label="D1 alarm")
    axes[0].axvspan(s0, s1, color=PAL[7], alpha=0.07, lw=0)
    axes[0].set_ylabel("kWh / 30 min"); axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    axes[0].set_title(f"Meter {m}: false data injection, alpha = 0.55, 3 days (shaded)")
    axes[1].plot(tt, z[m, a:b], color=PAL[0], lw=1.0)
    axes[1].axhline(-3, color=PAL[7], ls="--", lw=1); axes[1].axhline(3, color=PAL[7], ls="--", lw=1)
    axes[1].axvspan(s0, s1, color=PAL[7], alpha=0.07, lw=0)
    axes[1].set_ylabel("z-score (Eq. 2)"); axes[1].set_ylim(-9, 6)
    day_axis(axes[1], a, b)
    fig.tight_layout(); save(fig, "fig17_d1_example.png")

    s3 = D.d3_value_scores(Xa, mu)
    g = m // C.GROUP_SIZE; rows = range(g * C.GROUP_SIZE, (g + 1) * C.GROUP_SIZE)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = Xa[list(rows), a:b] / mu[list(rows), a:b]
    for r in R:
        axes[0].plot(tt, r, color=GRID, lw=0.6)
    axes[0].plot(tt, np.nanmedian(R, axis=0), color=INK, lw=1.4, label="Group median ratio")
    axes[0].plot(tt, Xa[m, a:b] / mu[m, a:b], color=PAL[2], lw=1.3, label=f"Meter {m} (attacked)")
    axes[0].axvspan(s0, s1, color=PAL[7], alpha=0.07, lw=0)
    axes[0].set_ylim(0, 2.5); axes[0].set_ylabel("reading / own baseline")
    axes[0].legend(fontsize=8, loc="upper left"); axes[0].set_title(f"Neighbourhood group {g + 1}: 20 meters")
    axes[1].plot(tt, s3[m, a:b], color=PAL[2], lw=1.0)
    axes[1].axhline(-3.5, color=PAL[7], ls="--", lw=1)
    fl3 = (D.run_length(np.abs(np.nan_to_num(s3)) > C.D3_Z) >= C.D3_K)[m, a:b]
    axes[1].scatter(tt[fl3], s3[m, a:b][fl3], s=8, color=PAL[7], zorder=5, label="N-VALUE alarm")
    axes[1].axvspan(s0, s1, color=PAL[7], alpha=0.07, lw=0)
    axes[1].set_ylabel("neighbourhood score"); axes[1].set_ylim(-12, 6); axes[1].legend(fontsize=8)
    day_axis(axes[1], a, b)
    fig.tight_layout(); save(fig, "fig18_d3_example.png")

    # weather robustness: D1 vs D3 alarms during a heat/cold event
    wdays = np.flatnonzero((benign[0].reshape(C.N_DAYS, C.SLOTS) == 2).any(axis=1))
    wdays = wdays[wdays > C.WARMUP_DAYS]
    d0 = wdays[0]; a, b = (d0 - 3) * C.SLOTS, (d0 + 8) * C.SLOTS
    f3 = D.run_length(np.abs(np.nan_to_num(s3)) > C.D3_Z) >= C.D3_K
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(np.arange(a, b), 100 * f1[:, a:b].mean(axis=0), color=PAL[0], lw=1.2, label="D1: % of meters alarmed")
    ax.plot(np.arange(a, b), 100 * f3[:, a:b].mean(axis=0), color=PAL[2], lw=1.2, label="D3 N-VALUE: % of meters alarmed")
    wmask = (benign[0, a:b] == 2)
    ax.fill_between(np.arange(a, b), 0, 100, where=wmask, color=PAL[3], alpha=0.15, lw=0, label="Weather event (benign)")
    ax.set_ylim(0, max(5, 100 * f1[:, a:b].mean(axis=0).max() * 1.2)); ax.set_ylabel("% of 500 meters")
    day_axis(ax, a, b, 2); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); save(fig, "fig19_weather_false_alarms.png")

    att = attacks.inject(data, "S1", np.random.default_rng(6), fixed={"alpha": 0.8, "days": 4, "fraction": 0.05})
    Xa = att["X"]; z, mu = D.d1_scores(Xa); s3 = D.d3_value_scores(Xa, mu)
    e = att["events"][0]; m = e["meter"]; s0, s1 = e["start"], e["end"]
    a, b = s0 - 2 * C.SLOTS, s1 + 2 * C.SLOTS; tt = np.arange(a, b)
    zc = np.clip(np.nan_to_num(-s3[m:m + 1], nan=1.0), -6, 6)
    S = np.zeros(1); Sv = []
    for t in range(C.T):
        S = np.minimum(np.maximum(0, S + zc[:, t] - 1.0), 30.0); Sv.append(S[0])
    Sv = np.array(Sv)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    axes[0].plot(tt, s3[m, a:b], color=PAL[2], lw=0.9, label="Neighbourhood score")
    axes[0].axhline(-3.5, color=PAL[7], ls="--", lw=1, label="D3 single-reading threshold")
    axes[0].axvspan(s0, s1, color=PAL[7], alpha=0.07, lw=0); axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_title(f"Meter {m}: small injection, alpha = 0.8 (shaded)"); axes[0].set_ylabel("score")
    axes[1].plot(tt, Sv[a:b], color=PAL[3], lw=1.4, label="CUSUM statistic S_t (cap 30)")
    axes[1].axhline(20, color=PAL[7], ls="--", lw=1, label="Alarm threshold h = 20")
    axes[1].axvspan(s0, s1, color=PAL[7], alpha=0.07, lw=0); axes[1].legend(fontsize=8, loc="upper left")
    axes[1].set_ylabel("S_t"); day_axis(axes[1], a, b)
    fig.tight_layout(); save(fig, "fig20_cusum_example.png")


# =============================================================== results
def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return np.nan
    return 2.045 * x.std(ddof=1) / np.sqrt(len(x))


def results():
    df = pd.read_csv(f"{RES}/main.csv")
    agg = df.groupby(["scenario", "method"]).agg(
        DR=("DR", "mean"), DR_ci=("DR", ci95), FPR=("FPR", "mean"), FPR_ci=("FPR", ci95),
        evDR=("event_DR", "mean"), evDR_ci=("event_DR", ci95),
        delay=("delay", "mean"), delay_ci=("delay", ci95)).reset_index()
    agg.to_csv(f"{RES}/summary_main.csv", index=False)
    scen = ["S1", "S2", "S3", "S4"]
    snames = ["S1 FDI", "S2 Interruption", "S3 Replay/freeze", "S4 Status burst"]

    def grouped(metric, ci, ylabel, fname, methods=METHODS, ylim=None, clip=None):
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        w = 0.8 / len(methods)
        for j, mth in enumerate(methods):
            sub = agg[agg.method == mth].set_index("scenario").loc[scen]
            x = np.arange(4) + (j - (len(methods) - 1) / 2) * w
            ax.bar(x, sub[metric], width=w * 0.9, color=MCOL[mth], label=LABEL[mth],
                   yerr=sub[ci], error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        ax.set_xticks(range(4)); ax.set_xticklabels(snames); ax.set_ylabel(ylabel)
        if ylim: ax.set_ylim(*ylim)
        if clip:
            ax.set_ylim(0, clip)
            for j, mth in enumerate(methods):
                sub = agg[agg.method == mth].set_index("scenario").loc[scen]
                for i, v in enumerate(sub[metric]):
                    if v > clip:
                        x = i + (j - (len(methods) - 1) / 2) * w
                        ax.text(x, clip * 0.97, f"{v:.0f}*", ha="center", va="top", fontsize=7.5, color="white", fontweight="bold")
            ax.text(0.99, 0.97, "* bar cut at axis limit; D2 finds only ~1% of S1 events", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, color=INK2)
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        fig.tight_layout(); save(fig, fname)

    grouped("DR", "DR_ci", "Detection rate (%)", "fig21_detection_rate.png", ylim=(0, 105))
    grouped("delay", "delay_ci", "Mean detection delay (30-min intervals)", "fig23_delay.png", clip=40)

    # FPR + false-positive sources
    fp_cols = ["FP_vacation", "FP_weather", "FP_legit_off", "FP_comms", "FP_none"]
    fpn = ["Vacation", "Weather event", "Legitimate disconnection", "Benign comms fault", "Normal data"]
    sub = df.groupby("method")[fp_cols + ["FP", "TN"]].sum()
    sub = sub.loc[METHODS]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})
    f = df.groupby("method")["FPR"].agg(["mean", ci95]).loc[METHODS]
    axes[0].barh(range(len(METHODS)), f["mean"], xerr=f["ci95"], color=[MCOL[m] for m in METHODS],
                 error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    for i, (v, c) in enumerate(zip(f["mean"], f["ci95"])):
        axes[0].text(v + c + 0.04, i, f"{v:.2f}%", va="center", fontsize=8, color=INK)
    axes[0].set_yticks(range(len(METHODS))); axes[0].set_yticklabels([LABEL[m] for m in METHODS], fontsize=8)
    axes[0].invert_yaxis(); axes[0].set_xlabel("False positive rate (%)"); axes[0].set_title("(a) FPR, all scenarios")
    axes[0].set_xlim(0, f["mean"].max() * 1.35); axes[0].grid(axis="y", visible=False)
    share = sub[fp_cols].div(sub[fp_cols].sum(axis=1), axis=0) * 100
    left = np.zeros(len(METHODS))
    src_col = [PAL[4], PAL[3], PAL[1], PAL[6], "#9a9890"]
    for k, c in enumerate(fp_cols):
        axes[1].barh(range(len(METHODS)), share[c], left=left, color=src_col[k], label=fpn[k],
                     edgecolor="white", linewidth=1.0)
        left += share[c].values
    axes[1].set_yticks(range(len(METHODS))); axes[1].set_yticklabels([])
    axes[1].invert_yaxis(); axes[1].set_xlabel("Share of false positives (%)"); axes[1].set_title("(b) What caused the false alarms")
    axes[1].set_xlim(0, 100); axes[1].grid(axis="y", visible=False)
    axes[1].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout(); save(fig, "fig22_false_positives.png")

    # event-level heatmap by variant
    var = pd.read_csv(f"{RES}/main_variants.csv")
    var["case"] = var["scenario"] + " " + var["variant"].replace("-", "")
    hm = var.groupby(["case", "method"])["event_DR"].mean().unstack().loc[:, METHODS]
    order = ["S1 ", "S2 single", "S2 group", "S3 freeze", "S3 replay", "S4 "]
    hm = hm.loc[[o for o in order if o in hm.index]]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#a9c9f0", "#2a78d6", "#0d3d7a"])
    im = ax.imshow(hm.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 60 else INK)
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([LABEL[m].replace(" (extension)", "") for m in METHODS], fontsize=8, rotation=20, ha="right")
    ylab = {"S1 ": "S1 FDI", "S2 single": "S2 single meter", "S2 group": "S2 group", "S3 freeze": "S3 freeze",
            "S3 replay": "S3 replay", "S4 ": "S4 status burst"}
    ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels([ylab[i] for i in hm.index], fontsize=9)
    ax.grid(False); cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("Events detected (%)")
    fig.tight_layout(); save(fig, "fig24_event_heatmap.png")
    hm.to_csv(f"{RES}/summary_variants.csv")

    # sensitivity sweeps
    sw = pd.read_csv(f"{RES}/sweep.csv")
    sw.groupby(["scenario", "value", "method"])[["DR", "FPR", "event_DR", "delay"]].mean().to_csv(f"{RES}/summary_sweep.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8))
    xl = {"S1": "Scale factor alpha (lower = larger theft)", "S2": "Gap duration (hours)",
          "S3": "Freeze / replay duration (hours)", "S4": "Fraction of meters switched off (%)"}
    for ax, sc in zip(axes.ravel(), scen):
        s = sw[sw.scenario == sc]
        for mth in ["D1", "D2", "D3", "E1", "Combined+E1"]:
            g = s[s.method == mth].groupby("value")["DR"].agg(["mean", ci95])
            x = g.index * (100 if sc == "S4" else 1)
            ax.plot(x, g["mean"], marker="o", ms=4, color=MCOL[mth], label=LABEL[mth])
            ax.fill_between(x, g["mean"] - g["ci95"], g["mean"] + g["ci95"], color=MCOL[mth], alpha=0.12, lw=0)
        ax.set_title(snames[scen.index(sc)], fontsize=10); ax.set_xlabel(xl[sc], fontsize=9)
        ax.set_ylabel("Detection rate (%)", fontsize=9); ax.set_ylim(-3, 103)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "fig25_sensitivity.png")

    # threshold trade-off
    th = pd.read_csv(f"{RES}/threshold.csv")
    th.groupby(["method", "thr"])[["DR", "FPR", "delay"]].mean().to_csv(f"{RES}/summary_threshold.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for mth, col in [("D1", PAL[0]), ("D3 (N-VALUE)", PAL[2]), ("E1 (CUSUM)", PAL[3])]:
        g = th[th.method == mth].groupby("thr")[["DR", "FPR"]].mean()
        ax.plot(g["FPR"], g["DR"], marker="o", ms=5, color=col, label=mth)
        for thr, r in g.iterrows():
            ax.annotate(f"{thr:g}", (r["FPR"], r["DR"]), textcoords="offset points", xytext=(4, 3), fontsize=7, color=INK2)
    ax.set_xlabel("False positive rate (%)"); ax.set_ylabel("Detection rate on S1 (%)")
    ax.set_title("Threshold trade-off on false data injection (labels = threshold)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); save(fig, "fig26_threshold_tradeoff.png")


# =============================================================== diagrams
def box(ax, x, y, w, h, text, fc, ec=INK2, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0))


def diagrams():
    blue, green, yellow, pink, violet, grey = "#dbe8f8", "#d6f0e5", "#fbefcc", "#fae1ea", "#e3dff7", "#efeeea"
    # ---- Fig 12 simulation architecture
    fig, ax = plt.subplots(figsize=(7.4, 3.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    box(ax, 0.1, 3.6, 2.1, 1.2, "Configuration\n(config.py)\nTable 11 + A11–A14", grey)
    box(ax, 2.6, 3.6, 2.2, 1.2, "Load generator\nEq. (1): B·D·W·S·(1+ε)\n500 meters × 17,520", blue)
    box(ax, 5.2, 3.6, 2.2, 1.2, "Benign events\nvacations, weather,\noutages, disconnections", yellow)
    box(ax, 7.8, 3.6, 2.1, 1.2, "Clean dataset\nX (kWh), status,\nbenign mask", blue)
    box(ax, 7.8, 1.9, 2.1, 1.2, "Attack injector\nS1–S4, 8 campaigns\nper run (Table 15)", pink)
    box(ax, 5.2, 1.9, 2.2, 1.2, "Labelled test data\nper-reading label +\nattack-event list", pink)
    box(ax, 2.6, 1.9, 2.2, 1.2, "Detectors\nD1, D2, D3\n(+ E1 extension)", green)
    box(ax, 0.1, 1.9, 2.1, 1.2, "Evaluation\nDR, FPR, delay\nEqs. (5)–(7)", violet)
    box(ax, 0.1, 0.2, 9.8, 1.0, "Repeated: 30 seeds × 4 scenarios (main experiment)  ·  10 seeds × 20 settings (sensitivity)\n10 seeds × threshold grid (trade-off study)", grey, fs=8.5)
    for (x0, x1) in [(2.2, 2.6), (4.8, 5.2), (7.4, 7.8)]:
        arrow(ax, x0, 4.2, x1, 4.2)
    arrow(ax, 8.85, 3.6, 8.85, 3.1)
    for (x0, x1) in [(7.8, 7.4), (5.2, 4.8), (2.6, 2.2)]:
        arrow(ax, x0, 2.5, x1, 2.5)
    arrow(ax, 1.15, 1.9, 1.15, 1.2)
    save(fig, "fig12_simulation_architecture.png")

    # ---- Fig 16 detection architecture
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.6); ax.axis("off")
    box(ax, 0.1, 2.2, 1.8, 1.2, "Incoming reading\nx(i,t), status(i,t)\nfrom head-end", grey)
    box(ax, 2.5, 4.1, 3.4, 1.2, "D1  Temporal view (one meter)\nz = (x − μ)/σ, 28-day same-slot\nalarm: |z| > 3 for 3 readings", blue)
    box(ax, 2.5, 2.2, 3.4, 1.2, "D2  Integrity rules\nR-GAP, R-REPEAT, R-REPLAY,\nR-RANGE, R-BURST", "#fde4d6")
    box(ax, 2.5, 0.3, 3.4, 1.2, "D3  Spatial view (20 neighbours)\nN-VALUE, N-SILENT, N-STATUS\n(+ E1 CUSUM on N-score)", green)
    box(ax, 6.5, 2.2, 1.6, 1.2, "Alarm fusion\n(OR) + reason\ncodes", violet)
    box(ax, 8.5, 2.2, 1.4, 1.2, "Operator\nconsole /\nSOC ticket", grey)
    for y in (4.7, 2.8, 0.9):
        arrow(ax, 1.9, 2.8, 2.5, y); arrow(ax, 5.9, y, 6.5, 2.8)
    arrow(ax, 8.1, 2.8, 8.5, 2.8)
    ax.text(7.3, 4.5, "Every alarm names the\nstatistic or rule that fired\n(explainability)", ha="center", fontsize=8, color=INK2, style="italic")
    save(fig, "fig16_detection_architecture.png")

    # ---- Fig 27 defence in depth
    fig, ax = plt.subplots(figsize=(7.4, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    layers = [
        ("Layer 5  Governance & recovery", "R8 incident response plan · R9 offline backups, manual reading · exercises · supplier security", grey),
        ("Layer 4  Detection & monitoring", "R5 completeness/heartbeat · R6 D1+D2+D3+E1 · SOC triage · threat hunting", green),
        ("Layer 3  Command assurance", "R7 tamper-evident command log · rate limits · two-person rule for bulk disconnect", "#fde4d6"),
        ("Layer 2  Network & communication", "R3 zones/conduits, DMZ · R4 authenticated + encrypted messages, counters vs replay", blue),
        ("Layer 1  Identity & device", "R1 MFA · R2 RBAC/least privilege · signed firmware · secure meter keys", violet),
    ]
    for k, (t, d, c) in enumerate(layers):
        inset = 0.25 * k
        y = 5.0 - k * 1.15
        box(ax, 0.1 + inset, y, 9.8 - 2 * inset, 1.0, "", c)
        ax.text(0.3 + inset, y + 0.68, t, fontsize=9.5, fontweight="bold", color=INK)
        ax.text(0.3 + inset, y + 0.28, d, fontsize=8.2, color=INK)
    save(fig, "fig27_defence_in_depth.png")

    # ---- Fig 28 roadmap
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    items = [("MFA + RBAC on head-end and vendor access (R1, R2)", 0, 3, 0),
             ("Command logging, rate limits, two-person bulk rule (R7)", 1, 4, 0),
             ("Deploy D2 rules + heartbeat monitoring (R5)", 0, 3, 1),
             ("Deploy D1 + D3 + E1 with tuning on utility data (R6)", 3, 9, 1),
             ("Network zoning, DMZ between MDMS and OT (R3)", 3, 12, 2),
             ("Authenticated/encrypted meter comms, key mgmt (R4)", 6, 24, 2),
             ("Incident response plan + tabletop exercise (R8)", 2, 6, 3),
             ("Offline backups, manual-reading drill (R9)", 4, 10, 3)]
    cols = [PAL[0], PAL[2], PAL[6], PAL[1]]
    for i, (name, s, e, c) in enumerate(items):
        ax.barh(i, e - s, left=s, color=cols[c], height=0.55)
        ax.text(e + 0.3, i, name, va="center", fontsize=7.8, color=INK)
    ax.set_yticks([]); ax.invert_yaxis(); ax.set_xlim(0, 40); ax.set_xlabel("Months from start")
    ax.set_xticks([0, 3, 6, 12, 18, 24])
    ax.legend(handles=[Patch(color=cols[0], label="Quick wins - access"), Patch(color=cols[1], label="Monitoring & detection"),
                       Patch(color=cols[2], label="Architecture"), Patch(color=cols[3], label="Response & recovery")],
              fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, "fig28_roadmap.png")

    # ---- Fig 29 cost vs risk reduction (qualitative, author assessment)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    pts = [("R1 MFA", 1.5, 4.5), ("R2 RBAC", 2.0, 3.8), ("R3 Segmentation", 3.5, 4.0), ("R4 Authenticated comms", 4.6, 4.4),
           ("R5 Monitoring", 1.8, 3.4), ("R6 Anomaly detection", 2.4, 3.6), ("R7 Command limits", 1.4, 4.2),
           ("R8 Incident response", 2.2, 3.0), ("R9 Offline recovery", 3.0, 3.2)]
    ax.axhspan(3.75, 5, xmin=0, xmax=0.5, color="#d6f0e5", alpha=0.6, lw=0)
    ax.text(0.65, 4.85, "quick wins", fontsize=8, color=INK2, style="italic")
    for name, x, y in pts:
        ax.scatter(x, y, s=60, color=PAL[0], edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK)
    ax.set_xlim(0.5, 6.4); ax.set_ylim(2.5, 5)
    ax.set_xlabel("Relative cost and implementation effort (1 low – 5 high)")
    ax.set_ylabel("Risk reduction against\nscenarios S1–S4 (author estimate)")
    fig.tight_layout(); save(fig, "fig29_cost_benefit.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("diagrams", "all"): diagrams()
    if what in ("illustrative", "all"): illustrative()
    if what in ("results", "all"): results()
