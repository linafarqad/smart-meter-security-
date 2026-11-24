import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "STIXGeneral"})
EQ = {
 8: r"$D_i(h) = c_i\left[\,0.45 + a_i\, e^{-(h-8-s_i)^2/2} + b_i\, e^{-(h-19-s_i)^2/(2\cdot 1.6^2)} + 0.25\, e^{-(h-13)^2/(2\cdot 3^2)}\right]$",
 9: r"$W(d) = \{1.10$ weekend$,\; 1.00$ weekday$\},\qquad S(d) = 1 + 0.20\,\cos\!\left(2\pi\,\frac{d-15}{365}\right)$",
 10: r"$y_{i,t} = 1$ if $x'_{i,t} \neq x_{i,t}$ because of an attack, else $0;\qquad e = (i,\ t_{\mathrm{start}},\ t_{\mathrm{end}})$",
 11: r"$\tilde{\sigma}_{i,t} = \max\left(\sigma_{i,t},\ 0.05\,\mu_{i,t} + 0.01\right)$",
 12: r"$\mathrm{alarm}_{i,t} = 1 \quad$ if $\quad |z_{i,t-j}| > \tau_1 \quad$ for all $\ j = 0, 1, \ldots, k_1 - 1$",
 13: r"$r_{i,t} = \dfrac{x_{i,t}}{\mu_{i,t}}$",
 14: r"$n_{i,t} = \dfrac{r_{i,t} - m_{g,t}}{\max\left(1.4826\cdot \mathrm{MAD}_{g,t},\ 0.05\right)}$",
 15: r"$S_{i,t} = \min\left(\max\left(0,\ S_{i,t-1} - n_{i,t} - \kappa\right),\ 1.5\,h\right), \qquad$ alarm if $S_{i,t} > h$",
 16: r"$\mathrm{CI}_{95} = \bar{x} \pm t_{0.975,\,n-1}\,\dfrac{s}{\sqrt{n}}$",
}
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eq"), exist_ok=True)
for n, tex in EQ.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=12)
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eq", f"eq{n}.png"), dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=False, facecolor="white")
    plt.close(fig)
print("ok")
