# Experiment code: which file is which experiment

Everything was run on Kaggle (2×T4), each script as a single notebook cell. The docstring at the top
of each file describes the experiment; the `NOTEBOOK_ARGV` line near the end shows the exact settings
it was run with on Kaggle.

| File | Experiment | Where it appears in the proposal |
|---|---|---|
| `span_knockout_harness.py` | Study 1: chain-free accuracy (GPQA 28.3%, MMLU 59.4%). Study 2: attention knockout on the hint sentence and letter rewriting on OLMo (0/75, 48/75). The Study 2 numbers are superseded by Study 3. | First emails to Kerem |
| `tokenizer_diagnostic_cell.py` | Diagnoses why Study 2 and Study 3 produced different token counts on the same chains (tokenizer version). | Background |
| `study3_tier1_data_cell.py` | Results of Study 3's first session, embedded in the code. Run BEFORE `study3_mediation.py`. | — |
| `study3_mediation.py` | Study 3: on OLMo, attention knockout, placebo, rewriting the letter only in the sentence vs. everywhere (cr_span, cr_all ≈ 90%, cr_swap, cr_consistent). | 4.4 (earlier intervention study) |
| `study4_protocol.py` (v1.3) | Study 4: the CMA protocol of 2512.23032; original and letter-level readout, symmetric NDE, strict filter, Filler Tokens, R3 (chain truncation), R4 (letter vs. option-text swap). For OLMo, and for Qwen with `--model qwen3-8b`. | 4.1, 4.2, 4.4 |
| `study5_kerem_setup.py` | Study 5: Kerem's own setup on Llama-3.2-3B and Llama-3-8B; three readouts, BCa confidence intervals, match to Figure 8 and Table 1. | Llama rows in 4.1 and 4.2 |
| `study6_qwen_generate.py` | Study 6: generating Qwen3-8B chains with Young's protocol (vLLM). | Qwen rows in 4.1, 4.2, 4.4 |
| `resume/study6_resume_data_{1..4}of4.py` | Embedded data (Qwen chains) for Study 6's second session. Run in order, before `study6_qwen_generate.py` and `study4_protocol.py`. | — |
| `study7_black_squares.py` | Study 7: Black Squares, released code vs. ■ actually inserted; Llama-3.2-3B, 500 OpenbookQA questions, binomial test. | 4.3 |
| `r3_grafigi.py` | The R3 chart (matplotlib). | 4.4 chart |

Kaggle order (Qwen session): resume 1→4, then `study6_qwen_generate.py` (`--engine none`), then `study4_protocol.py`.
