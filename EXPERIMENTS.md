# Experiment code: which file is which experiment

Everything ran on Kaggle (2× T4), each script as a single notebook cell. The top of each file has a
docstring describing the experiment. The `NOTEBOOK_ARGV` line near the end of each file shows the
settings used on Kaggle; for all files below it is `"all"`. Every run writes a `prereg_*.json` with the
code hash and the predictions before any data, so the version that ran can be checked against the hash
in `results/`.

| File | Version | Experiment | Results |
|---|---|---|---|
| `study8_threat_probe.py` | study8/0.2 | Study 8 on Qwen3-8B with thinking off: rule-based simulation of the published diagnostics; LLM games (natA/B/Info, trigger, miscal, myopic); final/bot/blind probes, punishment probe P, threat share S; L&P benchmark check. The pilot ran v0.1 of this file with `--pilot` (`results/study08/pilot/`); v0.2's amendments are listed in the file. | `results/study08/` |
| `study9_cot_carriers.py` | study9/0.2.1 | Study 9: symmetric sentence knockout vs. length-matched placebo, implants, on-policy resampling, keyword and LLM monitors. Study 8's 420 decisions are embedded. | `results/study09/` (incl. 50-sentence `labels_for_humans.csv`) |
| `study9b_numbers.py` | study9b/0.1 | Study 9b: empty chain; previous price shifted in the chain vs. in the prompt; counter-argument sentence. | `results/study09b/` |
| `analysis/cluster.py` | — | Agent- and game-cluster bootstrap re-analysis of Studies 8–9. | printed |
| `study10_replication.py` | study10/0.1 | Study 10: Studies 8/9/9b replicated on Qwen2.5-7B, Llama-3.1-8B, OLMo-2-7B (one model at a time; HF cache dropped between models). | `results/study10/` |
| `study11_12.py` | study11_12/0.1 | Study 12: incrimination evidence on organisms. 2,000 Study 8 decisions + 3,360 resamples + interrogation, four judges; calibration matrix, Ctrl-Z resampling and "≥6 of 8" rule, Wen macro-protocol. Also Study 11 v0.1 (thinking on, failed quality gate, exploratory). | `results/study11_12/` |
| `study11_v2.py` | study11/0.2 | Study 11 v0.2: Qwen3-8B with thinking on, budget forcing, `quiet` organism, four judges, full-trace and action-only views. | `results/study11_v2/` |
| `study13.py` | study13/0.1 | Study 13: v0.2 traces re-scored at excerpt level (embedded), new games with `quietT`, 6 agents per group, keyword and length baselines. | `results/study13/` |
| `study14a.py` | study14/0.2 | Study 14 run A: teacher data; Qwen3-8B prompted games incl. `quietL` and 8 trigger + 8 myopic; three judges (Llama-3.1-8B, Qwen3-8B, Qwen2.5-14B-AWQ). | `results/study14/runA/` |
| `study14b.py` | study14/0.2 | Study 14 run B: DeepSeek-R1-Distill-Qwen-7B games and three judges. | `results/study14/runB/` |
| `study14c.py` | study14/0.3 | Run C: miscal teacher data; R1 activation features (R1 traces embedded). The LoRA training ran out of memory here; the errors are in `runC/lora_*_error.txt`. | `results/study14/runC/` |
| `study14d.py` | study14/0.3 | Run D: q3 activation features (q3 traces embedded). Split from C because of Kaggle's 1 MB notebook-source limit. | `results/study14/runD/` |
| `study14e.py` | study14/0.3 | Run E: QLoRA training (fp16 frozen weights, chunked loss), trainedH and trainedM games, their activations, three judges. Both teacher files embedded. Also writes the joint A–E report (`report` command). | `results/study14/runE/`, `results/study14/joint_AtoE/` |
| `analysis/probe_check.py` | — | Activation probe vs. bag-of-words control on all runs; controls for secret words, length and prompt echo. | printed (numbers in README) |
| `figures/make_monitor_matrix.py` | — | The monitor × organism figure. | `figures/monitor_matrix.png` |
| `r3_chart_en.py` | — | English version of `update1/r3_grafigi.py` (R3 chart of the hint study, same data). | `figures/R3_chart_en.png` |
| `superseded/study12_incrimination.py` | 0.1 | The standalone Study 12; merged into `study11_12.py` before it ran. Never run. | — |

Kaggle order for Study 14: run A and run B in any order, then C, D and E. The joint report can be
rebuilt without a GPU (see README, "How to run").

The sha256 prefixes recorded in the reports are ef92159f5cc49e98 (14a), cef7952230554446 (14b),
648096f5e6723061 (14c), ff525b2953ce88ba (14d) and 3be6d90f4c2007c2 (14e).
