| measure | Qwen3-8B (S8-9b) | Qwen2.5-7B-Instruct | Meta-Llama-3.1-8B-Instru | OLMo-2-1124-7B-Instruct |
|---|---|---|---|---|
| natA price, last 20 rounds | 3.329 | 1.701 | 1.656 | 1.721 |
| natB price, last 20 rounds | 1.618 | 0.886 | 0.237 | 0.218 |
| myopic price, last 20 rounds | 3.576 | 2.335 | 2.890 | 2.004 |
| trigger price, last 20 rounds | 1.900 | 1.900 | 1.895 | 1.899 |
| natA Delta (ours) | -1.274 | 0.502 | -0.412 | -0.056 |
| myopic rival-lag coef | 0.288 | 0.421 | 0.201 | 0.187 |
| myopic rival-lag p | 5.6e-08 | 5.8e-15 | 7.8e-05 | 1.9e-05 |
| natA rival-lag coef | 0.291 | 0.319 | 0.296 | 0.158 |
| trigger R_final | +0.064 [+0.056,+0.073] | -0.031 [-0.161,+0.099] | +0.052 [+0.004,+0.100] | +0.609 [+0.404,+0.815] |
| myopic R_final | -0.020 [-0.058,+0.018] | -0.152 [-0.237,-0.068] | -0.070 [-0.119,-0.021] | -0.043 [-0.115,+0.028] |
| natA R_final | -0.025 [-0.039,-0.010] | -0.016 [-0.042,+0.009] | -0.045 [-0.073,-0.017] | -0.031 [-0.087,+0.024] |
| natA S (threat share) | -0.003 [-0.011,+0.005] | +0.587 [+0.477,+0.698] | -0.007 [-0.033,+0.019] | +0.160 [-0.067,+0.388] |
| natA R_bot | n/a | -0.033 [-0.065,-0.002] | -0.104 [-0.136,-0.072] | -0.053 [-0.104,-0.002] |
| natA belief: curve rises | 82% | 31% | 15% | 18% |
| natA empty - full | -0.011 [-0.039,+0.017] | +0.014 [-0.007,+0.034] | -0.035 [-0.061,-0.010] | -0.008 [-0.054,+0.038] |
| natA T_chain | +0.517 [+0.385,+0.650] | +0.022 [-0.030,+0.074] | +0.043 [-0.038,+0.124] | +0.141 [+0.051,+0.231] |
| natA T_prompt | +0.306 [+0.207,+0.406] | +0.389 [+0.240,+0.538] | +0.354 [+0.248,+0.459] | +0.502 [+0.324,+0.679] |
| natA counter - full | -0.460 [-0.497,-0.423] | -0.095 [-0.127,-0.064] | -0.217 [-0.255,-0.179] | -0.138 [-0.166,-0.110] |
| keyword flag: myopic | 68% | 64% | 64% | 54% |
| keyword flag: miscal | 82% | 88% | 58% | 62% |
| judge flag: trigger | 100% | 100% | 70% | 75% |
| judge flag: miscal | 80% | 12% | 25% | 8% |
| judge flag: myopic | 35% | 0% | 4% | 0% |

| prediction | Qwen2.5-7B-Instruct | Meta-Llama-3.1-8B-Instru | OLMo-2-1124-7B-Instruct |
|---|---|---|---|
| R1_lag_flags_noncolluder | PASS | PASS | PASS |
| R2_placebo_null | PASS | PASS | PASS |
| R3_positive_control | FAIL | PASS | PASS |
| R4_natural_not_threat | not interpretable (R3 failed) | not estimable (20 states with p_base > p_N + 0.05) | not estimable (22 states with p_base > p_N + 0.05) |
| R5_chain_inert | PASS | PASS | PASS |
| R6_chain_numbers | FAIL | FAIL | PASS |
| R7_counter | PASS | PASS | PASS |
| R8_keyword_monitor | PASS | PASS | PASS |
| R9_promptB | PASS (-0.82) | PASS (-1.42) | PASS (-1.50) |
| Q_parse_gate | PASS | PASS | PASS |
