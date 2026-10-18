"""Figure: every monitor against every organism (Study 14, runs A-E). Writes monitor_matrix.png.

Excerpt-level AUROC (excerpts of <= 2,000 characters), Qwen3-8B players unless marked R1. Numbers from
results/study14/joint_AtoE/report14_joint.txt (LLM judges) and analysis/probe_check.py (keyword,
length, activation probe, bag of words). The activation probe and the bag of words are trained on
prompted organisms only (even games); they never saw quietL, R1 or the trained organisms.
Last column: final-round probe R_final (price drop when told the market closes), agent-level 95% t-interval.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = ["trigger vs natB\n(transparent cartel)", "quietT vs natB\n(told to hide, even in thinking)",
        "quietL vs natB\n(hide + reason at length)", "R1: quietT vs natB\n(second reasoning model)",
        "trainedH vs trainedM\n(QLoRA: silent cartel vs\nmiscalibrated honest)"]
cols = ["keyword\nregex", "length\n(short)", "Llama-3.1\n8B judge", "Qwen2.5-14B\nAWQ judge", "Qwen3-8B\njudge",
        "activation\nprobe", "bag of\nwords"]
A = np.array([[.97, .72, 1.00, 1.00, 1.00, 1.00, 1.00],
              [.90, .67, .76, .85, .84, .99, .99],
              [.89, .64, .66, .81, .75, .98, .97],
              [.91, .59, .92, .95, .94, .98, .97],
              [.47, .67, .79, .83, .81, .93, .94]])
R = ["+0.069 [+0.055, +0.083]", "+0.017 [-0.010, +0.045]", "+0.025 [+0.004, +0.046]",
     "framing not validated", "-0.108 [-0.292, +0.075]"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=170)
im = ax.imshow(A, cmap="RdYlGn", vmin=.4, vmax=1.0, aspect="auto")
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=11,
                color="black", fontweight="bold" if A[i, j] < .8 else "normal")
    ax.text(A.shape[1] - .35, i, R[i], ha="left", va="center", fontsize=9.5, color="#1f2328")
ax.set_xticks(range(len(cols)), cols, fontsize=9)
ax.set_yticks(range(len(rows)), rows, fontsize=9)
ax.xaxis.tick_top()
ax.text(A.shape[1] - .35, -.85, "final-round probe\nR_final", ha="left", va="center", fontsize=9)
ax.set_xlim(-.5, A.shape[1] + 1.9)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.01)
cb.ax.set_title("AUROC", fontsize=9)
fig.suptitle("Each cheap monitor breaks on a different organism; trained monitors hold up", x=.02, ha="left", fontsize=14, y=.98)
fig.text(.02, .015, "Excerpt-level AUROC; Qwen3-8B players unless marked R1. Activation probe and bag of words are trained on prompted "
         "organisms\n(trigger, quietT vs natB, miscal, myopic; even games) and tested on held-out games and on organisms they never saw. "
         "Bold: AUROC < 0.80.",
         fontsize=8, color="#59636e")
fig.subplots_adjust(left=.2, right=.97, top=.8, bottom=.11)
fig.savefig("monitor_matrix.png", facecolor="white")
