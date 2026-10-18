"""R3 chart (English version of update1/r3_grafigi.py; same data).

Hint-free prompt + the first fraction of the hinted chain of thought; letter-level readout.
OLMo-3-7B-Think n=60, Qwen3-8B n=50 per hint type; means. Writes R3_chart_en.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f = [0, .25, .5, .75, 1]
S = {("OLMo-3-7B-Think", "sycophancy"): [.276, .530, .675, .814, .899],
     ("OLMo-3-7B-Think", "grader"): [.281, .609, .692, .788, .916],
     ("Qwen3-8B", "sycophancy"): [.087, .256, .310, .352, .591],
     ("Qwen3-8B", "grader"): [.085, .262, .318, .325, .697]}
GREY = "#8a8f98"; BLUE = "#2a6fdb"; INK = "#1f2328"; INK2 = "#59636e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
for (m, h), v in S.items():
    c = BLUE if m.startswith("Qwen") else GREY
    ax.plot(f, v, color=c, lw=2, ls="--" if h == "grader" else "-", marker="o", ms=6,
            markeredgecolor="white", markeredgewidth=1.5, label=f"{m}, {h}")
ax.axhline(.5, color="#c9ced6", lw=1, ls=":", zorder=0)
ax.set_xticks(f, ["no chain", "25%", "50%", "75%", "full chain"])
ax.set_yticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1"])
ax.set_ylim(0, 1.02); ax.set_xlim(-.07, 1.07)
ax.set_xlabel("Share of the hinted chain of thought given to the hint-free prompt", color=INK2)
ax.set_ylabel("P(hint letter)", color=INK2)
ax.grid(axis="y", color="#e6e8eb", lw=.8); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color("#c9ced6")
ax.tick_params(colors=INK2, length=0)
fig.suptitle("OLMo commits to the hinted answer early in the chain, Qwen only at the end",
             x=.06, ha="left", fontsize=14, color=INK, y=.97)
fig.text(.06, .885, "Hint-free prompt + the first part of the hinted chain of thought; letter-level readout; "
         "OLMo n=60, Qwen3-8B n=50 (per hint), means", fontsize=9, color=INK2)
ax.legend(frameon=False, loc="lower right", fontsize=9.5, labelcolor=INK, handlelength=4.5, markerscale=0.8)
fig.subplots_adjust(left=.09, right=.97, top=.84, bottom=.13)
fig.savefig("R3_chart_en.png", facecolor="white")
