# cot-hint-span-causality

Model: olmo-3-7b-think.
498 questions, released traces and stage 1 patterns (Richard J. Young).

chain-free accuracy = 28.3% GPQA (at chance, p=0.32) x 59.4% MMLU
knockout on the acknowledgement span = 0/75 answers moved
same traces, same span, letter rewritten = 48/75 moved

  Hiding the span does nothing, rewriting the letter it names moves two thirds. The null is not the intervention failing to bite (in 85-90% of these chains the letter is already restated elsewhere). Masking does not separate sycophancy from grader (+0.003), corruption does (-0.115).

  The classifier moves from the verdict to the span, so we can say not classifier-free. A  second regex independently matches the same sentence 95% of the time for grader, 20% for sycophancy.

  One model (nemotron-nano-9b ignores a 4D attention mask, so the measure does not exist on that architecture). Eligibility (27.5%/6.2%), mostly capped by GPU memory. Fidelity (87%/97%).

  Numbers: results/report_olmo3_7b_think.md
  Was edited in each chain: results/study2_span_edits_olmo3_partial.jsonl
  Per-condition scores: results/study2_condition_scores_olmo3.jsonl
  Harness: span_knockout_harness.py

  Also in results/-, trace eligibility, and an earlier run with all twelve conditions at n=14/12
