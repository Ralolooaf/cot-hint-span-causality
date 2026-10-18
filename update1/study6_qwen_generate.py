#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
study6_qwen_generate.py -- Study 6, part 1: reasoning chains from a second
reasoning model, Qwen3-8B, in Richard Young's format.
=============================================================================

Studies 3-4 rest on one model (OLMo-3-7B-Think) whose chains Young released.
For a second model there are no released chains, so this cell generates them
with Young's exact protocol, minus what an API run has and a T4 does not:

  questions  Young's 498 (data/sampled/mmlu_300.jsonl, gpqa_198.jsonl)
  prompt     his system prompt and his user messages, verbatim (the harness
             ports both): the unhinted question, and the sycophancy and grader
             hints
  target     the hinted letter Young drew for each question (the same letter
             for every hint type -- read from his released OLMo files)
  model      Qwen/Qwen3-8B, thinking mode, float16 (T4 has no bfloat16)
  sampling   temperature 0.6, top-p 0.95, top-k 20, one fixed seed per prompt.
             Young used temperature 0; Qwen's model card warns that greedy
             decoding makes Qwen3's thinking mode loop, so its recommended
             sampling is used instead. Every chain is generated once.
  length     up to 8192 tokens of context; a chain that does not close its
             </think> is kept but marked truncated and is never analysed

Output, in Young's schema, where the next cell (study4_protocol.py v1.2) reads
it: rjy_data/base/qwen3-8b.jsonl and rjy_data/hinted/qwen3-8b/{hint}.jsonl,
plus rjy_runs/study6_traces.zip so the chains outlive the session.

Engine: vLLM (installed into its own folder so it cannot disturb the notebook's
packages; tensor-parallel over both T4s), falling back to plain transformers
generation if vLLM cannot start. The fallback is several times slower; the
time budget then simply leaves the later questions ungenerated.

USAGE: paste this whole file into ONE Kaggle cell (GPU T4 x2, Internet on)
BEFORE the cell holding study4_protocol.py v1.2, then Save & Run All.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSION = "study6/1.0"
_IN_FILE = "__file__" in globals()
_T0 = time.time()
HARNESS_URL = ("https://raw.githubusercontent.com/Ralolooaf/cot-hint-span-causality/"
               "main/span_knockout_harness.py")
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
VLLM_PIN = "vllm==0.9.2"
# vLLM 0.9.2 does not cap transformers, and newer transformers (>= 4.54) already
# define a config vLLM registers itself ("aimv2"), so `from vllm import LLM`
# fails. 4.53.2 is the version vLLM 0.9.2 was released against (it has Qwen3).
TRANSFORMERS_PIN = "transformers==4.53.2"
# decoded WITH special tokens, so <think>/</think> survive whatever the tokenizer
# calls special; these end-of-turn markers are then removed
END_TOKENS = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def code_hash() -> str:
    try:
        return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()[:16]
    except Exception:
        return "unknown"


def load_harness():
    p = Path(os.environ.get("S3_HARNESS", "span_knockout_harness.py"))
    if not p.exists() or p.stat().st_size < 10000:
        log(f"downloading harness {HARNESS_URL}")
        with urllib.request.urlopen(HARNESS_URL, timeout=120) as r:
            p.write_bytes(r.read())
    spec = importlib.util.spec_from_file_location("rjy_harness", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rjy_harness"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    out = []
    if p.exists():
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


# =============================================================================
# 1. jobs
# =============================================================================

def build_jobs(a, H) -> List[Dict[str, Any]]:
    qs = H.load_questions_default(a)
    targets: Dict[str, str] = {}
    for h in a.hints:
        for r in H.load_traces(a, a.target_model, h):
            T = H.target_from_trace(r)
            qid = r.get("question_id")
            if qid in qs and T:
                if qid in targets and targets[qid] != T:
                    log(f"  note: {qid} has different targets across hints; using the first")
                targets.setdefault(qid, T)
    qids = sorted(targets, key=lambda q: (qs[q].dataset != "mmlu", _seed(q)))
    if a.max_questions:
        qids = qids[:a.max_questions]
    jobs = []
    for qid in qids:
        q, T = qs[qid], targets[qid]
        for h in [None] + list(a.hints):
            user = q.formatted if h is None else H.hinted_user_message(q, h, T)
            jobs.append({"job_id": f"{qid}|{h or 'base'}", "question_id": qid, "hint": h,
                         "target": T, "seed": _seed(qid, h or "base", "study6"),
                         "messages": [{"role": "system", "content": H.SYSTEM_PROMPT},
                                      {"role": "user", "content": user}]})
    log(f"{len(qids)} questions with a target ({sum(qs[q].dataset == 'mmlu' for q in qids)} MMLU "
        f"first), {len(jobs)} prompts")
    return jobs


# =============================================================================
# 2. workers (run in their own process)
# =============================================================================

def _render(tok, msgs) -> str:
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def worker_vllm(a, jobs: List[Dict[str, Any]], out: Path) -> None:
    os.environ.setdefault("VLLM_USE_V1", "0")          # V0 + xformers runs on Turing
    import torch
    from vllm import LLM, SamplingParams
    import vllm
    n_gpu = max(1, torch.cuda.device_count())
    log(f"vLLM {vllm.__version__}, torch {torch.__version__}, {n_gpu} GPU(s), eager={a.eager}")
    llm = LLM(model=a.hf_id, dtype="half", tensor_parallel_size=n_gpu, max_model_len=a.max_model_len,
              gpu_memory_utilization=0.90, max_num_seqs=64, seed=0, enforce_eager=a.eager,
              trust_remote_code=True)
    tok = llm.get_tokenizer()
    t0, n_tok, n_done = time.time(), 0, 0
    for s in range(0, len(jobs), a.chunk):
        if time.time() > a.deadline:
            log("time budget reached; stopping generation")
            return
        part = jobs[s:s + a.chunk]
        prompts = [_render(tok, j["messages"]) for j in part]
        sps = []
        for j, pr in zip(part, prompts):
            room = a.max_model_len - len(tok(pr).input_ids) - 8
            sps.append(SamplingParams(max_tokens=max(256, room), seed=j["seed"],
                                      skip_special_tokens=False, **SAMPLING))
        outs = llm.generate(prompts, sps, use_tqdm=False)
        with open(out, "a", encoding="utf-8") as f:
            for j, o in zip(part, outs):
                c = o.outputs[0]
                n_tok += len(c.token_ids)
                f.write(json.dumps({"job_id": j["job_id"], "text": c.text, "n_tokens": len(c.token_ids),
                                    "finish_reason": c.finish_reason, "engine": f"vllm {vllm.__version__}"},
                                   ensure_ascii=False) + "\n")
        n_done += len(part)
        el = time.time() - t0
        log(f"  {n_done}/{len(jobs)} prompts, {n_tok / el:.0f} tok/s, "
            f"eta {(len(jobs) - n_done) * el / n_done / 60:.0f} min")


def worker_hf(a, jobs: List[Dict[str, Any]], out: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(a.hf_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw: Dict[str, Any] = {"device_map": "auto" if torch.cuda.is_available() else {"": "cpu"}}
    dt = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(a.hf_id, dtype=dt, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.hf_id, torch_dtype=dt, **kw)
    model.eval()
    dev = next(model.parameters()).device
    log(f"transformers {transformers.__version__} generation fallback on {dev}; "
        f"max {a.hf_max_new} new tokens")
    t0, n_tok, n_done = time.time(), 0, 0
    for s in range(0, len(jobs), a.hf_batch):
        if time.time() > a.deadline:
            log("time budget reached; stopping generation")
            return
        part = jobs[s:s + a.hf_batch]
        enc = tok([_render(tok, j["messages"]) for j in part], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        torch.manual_seed(part[0]["seed"])
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, max_new_tokens=a.hf_max_new,
                               pad_token_id=tok.pad_token_id, **SAMPLING)
        gen = g[:, enc.input_ids.shape[1]:]
        with open(out, "a", encoding="utf-8") as f:
            for j, row in zip(part, gen):
                ids = [int(t) for t in row if int(t) != tok.pad_token_id]
                text = tok.decode(ids, skip_special_tokens=False)
                ended = any(int(t) == tok.eos_token_id for t in row) or len(ids) < a.hf_max_new
                n_tok += len(ids)
                f.write(json.dumps({"job_id": j["job_id"], "text": text, "n_tokens": len(ids),
                                    "finish_reason": "stop" if ended else "length",
                                    "engine": f"transformers {transformers.__version__}"},
                                   ensure_ascii=False) + "\n")
        n_done += len(part)
        el = time.time() - t0
        log(f"  {n_done}/{len(jobs)} prompts, {n_tok / el:.0f} tok/s")


def cmd_worker(a) -> None:
    jobs = json.loads(Path(a.jobs).read_text())
    out = Path(a.out)
    done = {r["job_id"] for r in _read_jsonl(out)}
    jobs = [j for j in jobs if j["job_id"] not in done]
    log(f"worker ({a.engine}): {len(jobs)} prompts to generate, {len(done)} already done")
    if jobs:
        (worker_vllm if a.engine == "vllm" else worker_hf)(a, jobs, out)
    # vLLM can hang on shutdown; everything is on disk, so leave at once. The
    # parent then kills this process group, which includes vLLM's GPU workers.
    print("WORKER_FINISHED", flush=True)
    os._exit(0)


# =============================================================================
# 3. orchestration (the notebook process)
# =============================================================================

def _run(cmd: List[str], timeout: int = 2400) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def install_vllm(venv: Path) -> Optional[str]:
    """vLLM in its OWN Python 3.12 environment, made with uv. The notebook's
    Python (3.13 on Kaggle since 2026) has no wheel for the vLLM versions that
    still run on a T4 (V0 engine + xformers, <= 0.10.1), and installing vLLM
    into the notebook's environment would replace its torch. Returns the
    environment's python, or None."""
    py = venv / "bin" / "python"
    check = [str(py), "-c", "from vllm import LLM, SamplingParams; import vllm, torch, transformers; "
             "import vllm.model_executor.models.qwen3, vllm.attention.backends.xformers; "
             "print(vllm.__version__, torch.__version__, transformers.__version__)"]
    if py.exists() and _run(check, 600).returncode == 0:
        return str(py)
    log(f"making a Python 3.12 environment with {VLLM_PIN} at {venv} (uv; a few minutes) ...")
    steps = [[sys.executable, "-m", "pip", "install", "-q", "uv"],
             [sys.executable, "-m", "uv", "venv", "--python", "3.12", str(venv)],
             [sys.executable, "-m", "uv", "pip", "install", "--python", str(py), VLLM_PIN,
              TRANSFORMERS_PIN]]
    if py.exists():                       # a broken earlier attempt: start clean
        import shutil
        shutil.rmtree(venv, ignore_errors=True)
    for cmd in steps:
        try:
            r = _run(cmd)
        except subprocess.TimeoutExpired:
            log(f"  timed out: {' '.join(cmd[2:5])}")
            return None
        if r.returncode != 0:
            log(f"  failed: {' '.join(cmd[2:5])}: {(r.stderr or r.stdout)[-800:]}")
            return None
    r = _run(check, 600)
    if r.returncode != 0:
        log(f"  vLLM does not import: {(r.stderr or r.stdout)[-800:]}")
        return None
    log(f"  vLLM ready: {r.stdout.strip()}")
    return str(py)


def run_worker(a, engine: str, jobs_file: Path, raw: Path, extra_env: Dict[str, str],
               extra_args: List[str], python: Optional[str] = None) -> int:
    """Run a worker in its own process group, relay its output, and make sure it
    is gone afterwards: if it finished but does not exit within 2 minutes, or
    runs 15 minutes past the deadline, the whole group (vLLM's GPU workers too)
    is killed, so the GPUs are free for the next cell."""
    import queue
    import signal
    import threading
    me = Path(__file__).resolve() if _IN_FILE else Path.cwd() / "study6_qwen_generate.py"
    cmd = [python or sys.executable, "-u", str(me), "worker", "--engine", engine, "--jobs", str(jobs_file),
           "--out", str(raw), "--hf-id", a.hf_id, "--deadline", str(a.deadline),
           "--max-model-len", str(a.max_model_len), "--chunk", str(a.chunk),
           "--hf-batch", str(a.hf_batch), "--hf-max-new", str(a.hf_max_new), *extra_args]
    env = dict(os.environ, **extra_env)
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          bufsize=1, env=env, start_new_session=True)
    lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def pump() -> None:
        assert pr.stdout is not None
        for line in pr.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    tail: List[str] = []
    finished_at: Optional[float] = None
    hard_stop = a.deadline + 15 * 60
    while True:
        try:
            line = lines.get(timeout=5)
        except queue.Empty:
            line = ""
        if line is None:
            break
        if line:
            tail = (tail + [line])[-40:]
            if "WORKER_FINISHED" in line:
                finished_at = time.time()
            elif "it/s]" not in line and "Processed prompts" not in line:
                print(line, end="", flush=True)
        now = time.time()
        if (finished_at and now - finished_at > 120) or now > hard_stop:
            log(f"  {engine} worker did not exit; killing it")
            break
    rc = pr.poll()
    try:
        os.killpg(pr.pid, signal.SIGKILL)      # the worker and everything it started
    except (ProcessLookupError, PermissionError):
        pass
    try:
        rc = pr.wait(timeout=60) if rc is None else rc
    except subprocess.TimeoutExpired:
        rc = -9
    if finished_at:
        rc = 0
    time.sleep(10)                             # let the driver release the memory
    if rc != 0:
        log(f"  {engine} worker exited with {rc}; last lines:\n" + "".join(tail[-15:]))
    return rc


def parse_and_write(a, H, jobs: List[Dict[str, Any]], raw: Path) -> Dict[str, Any]:
    by = {r["job_id"]: r for r in _read_jsonl(raw)}
    data = Path(a.data_dir)
    rows: Dict[Optional[str], List[Dict[str, Any]]] = {None: []}
    for h in a.hints:
        rows[h] = []
    stats = {"generated": 0, "truncated": 0, "no_answer": 0}
    for j in jobs:
        g = by.get(j["job_id"])
        if not g:
            continue
        stats["generated"] += 1
        text = g["text"]
        for e in END_TOKENS:                  # kept by the decoder on purpose (see below)
            text = text.replace(e, "")
        text = text.rstrip()
        t = text.split("<think>", 1)[1] if "<think>" in text else text
        if "</think>" in t:
            think, ans = t.split("</think>", 1)
            trunc = g.get("finish_reason") == "length" and not ans.strip()
        else:
            think, ans, trunc = t, "", True
        ea = None if trunc else H.extract_answer(text, think, ans)
        stats["truncated"] += int(trunc)
        stats["no_answer"] += int(ea is None)
        rows[j["hint"]].append({
            "question_id": j["question_id"], "model_name": a.model, "hint_type": j["hint"],
            "target_label": j["target"] if j["hint"] else None,
            "thinking_text": think.strip(), "answer_text": ans.strip(), "full_response": text,
            "extracted_answer": ea, "truncated": trunc, "n_tokens": g.get("n_tokens"),
            "finish_reason": g.get("finish_reason"), "seed": j["seed"], "engine": g.get("engine"),
            "generator": VERSION})
    (data / "base").mkdir(parents=True, exist_ok=True)
    (data / "hinted" / a.model).mkdir(parents=True, exist_ok=True)
    for h, rs in rows.items():
        p = data / "base" / f"{a.model}.jsonl" if h is None else data / "hinted" / a.model / f"{h}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # influence, as Study 4 v1.2 will select it
    base = {r["question_id"]: r["extracted_answer"] for r in rows[None]}
    for h in a.hints:
        n_pair = sum(1 for r in rows[h] if r["question_id"] in base and not r["truncated"])
        n_inf = sum(1 for r in rows[h] if r["extracted_answer"] == r["target_label"]
                    and base.get(r["question_id"]) not in (None, r["target_label"]))
        stats[f"{h}_pairs"] = n_pair
        stats[f"{h}_influenced"] = n_inf
    return stats


SESSION_FILE = Path("rjy_runs") / "session_deadline.json"
SESSION_HOURS = 11.5        # Kaggle stops a session at 12 h; every cell ends before that


def write_session_deadline() -> None:
    """Written once, by the first cell of the session; every later run reads it."""
    if not SESSION_FILE.exists():
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(json.dumps({"deadline": time.time() + SESSION_HOURS * 3600,
                                            "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))


def session_end() -> Optional[float]:
    try:
        return float(json.loads(SESSION_FILE.read_text())["deadline"])
    except Exception:
        return None


def resume_generations(raw: Path) -> None:
    """New session: if an earlier session's output was added as input, take the
    longest raw_generations.jsonl found there (loose or inside study6_traces.zip),
    so only the missing prompts are generated."""
    import zipfile
    have = len(_read_jsonl(raw))
    root = Path("/kaggle/input")
    if not root.exists():
        return
    best, best_n = None, have
    for f in root.rglob("raw_generations.jsonl"):
        n = len(_read_jsonl(f))
        if n > best_n:
            best, best_n = f.read_text(encoding="utf-8"), n
    for z in root.rglob("study6_traces.zip"):
        with zipfile.ZipFile(z) as zf:
            if "raw_generations.jsonl" in zf.namelist():
                txt = zf.read("raw_generations.jsonl").decode("utf-8")
                n = sum(1 for line in txt.splitlines() if line.strip())
                if n > best_n:
                    best, best_n = txt, n
    if best is not None:
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(best, encoding="utf-8")
        log(f"resumed {best_n} generations from an earlier session's output")


def cmd_all(a) -> None:
    H = load_harness()
    H.cmd_fetch(argparse.Namespace(data_dir=a.data_dir, models=[a.target_model], hints=a.hints,
                                   force=False))
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    pre = run / "prereg_study6_generation.json"
    if not pre.exists():
        pre.write_text(json.dumps({"version": VERSION, "code_sha256_16": code_hash(),
                                   "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                   "model": a.hf_id, "sampling": SAMPLING, "max_model_len": a.max_model_len,
                                   "hints": a.hints, "targets_from": a.target_model,
                                   "system_prompt": H.SYSTEM_PROMPT}, indent=1))
    jobs = build_jobs(a, H)
    jobs_file = run / "jobs.json"
    jobs_file.write_text(json.dumps(jobs, ensure_ascii=False))
    raw = run / "raw_generations.jsonl"
    resume_generations(raw)
    a.deadline = _T0 + a.budget_hours * 3600
    end = session_end()
    if end:   # leave the session's remaining time to the analyses in the next cell
        a.deadline = min(a.deadline, end - a.reserve_hours * 3600)
        log(f"generation stops by {time.strftime('%H:%M', time.localtime(a.deadline))}; "
            f"session ends {time.strftime('%H:%M', time.localtime(end))}")
    left = lambda: len(jobs) - len({r["job_id"] for r in _read_jsonl(raw)})  # noqa: E731
    engines_tried = []
    if a.engine in ("auto", "vllm") and left():
        vpy = install_vllm(Path(a.vllm_dir))
        if vpy:
            env = {"VLLM_USE_V1": "0", "TOKENIZERS_PARALLELISM": "false",
                   "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
            for extra in ([], ["--eager"]):
                if not left() or time.time() > a.deadline:
                    break
                before = left()
                rc = run_worker(a, "vllm", jobs_file, raw, env, extra, python=vpy)
                engines_tried.append(f"vllm{' eager' if extra else ''}: rc={rc}")
                if rc == 0 or left() < before:
                    if left() and time.time() < a.deadline:   # crashed mid-way: resume once
                        rc = run_worker(a, "vllm", jobs_file, raw, env, extra, python=vpy)
                        engines_tried.append(f"vllm resume: rc={rc}")
                    break
    if a.engine in ("auto", "hf") and left() and time.time() < a.deadline:
        if a.engine == "auto":
            log("falling back to transformers generation")
        try:
            if not a.skip_deps:
                H.ensure_deps(upgrade=True)
        except Exception as e:
            log(f"  dependency check failed ({type(e).__name__}: {e}); trying anyway")
        rc = run_worker(a, "hf", jobs_file, raw, {}, [])
        engines_tried.append(f"hf: rc={rc}")
    stats = parse_and_write(a, H, jobs, raw)
    stats.update({"engines": engines_tried, "minutes": round((time.time() - _T0) / 60, 1)})
    (run / "generation_stats.json").write_text(json.dumps(stats, indent=1))
    log(f"generation stats: {stats}")
    import shutil
    pack = run / "pack"
    if pack.exists():
        shutil.rmtree(pack)
    (pack / "base").mkdir(parents=True)
    shutil.copy2(Path(a.data_dir) / "base" / f"{a.model}.jsonl", pack / "base")
    shutil.copytree(Path(a.data_dir) / "hinted" / a.model, pack / "hinted" / a.model)
    for f in ("prereg_study6_generation.json", "generation_stats.json", "raw_generations.jsonl"):
        if (run / f).exists():
            shutil.copy2(run / f, pack / f)
    shutil.make_archive(str(run.parent / "study6_traces"), "zip", root_dir=str(pack))
    shutil.rmtree(pack)
    log(f"packaged -> {run.parent / 'study6_traces.zip'}")
    if Path(a.vllm_dir).exists():          # ~6 GB the next cell does not need
        shutil.rmtree(a.vllm_dir, ignore_errors=True)
        log(f"removed {a.vllm_dir}")


def build_parser():
    p = argparse.ArgumentParser(prog="study6_qwen_generate.py")
    p.add_argument("cmd", choices=["all", "worker"])
    p.add_argument("--data-dir", default="rjy_data")
    p.add_argument("--run-dir", default="rjy_runs/study6_gen")
    p.add_argument("--model", default="qwen3-8b", help="name the chains are saved under")
    p.add_argument("--hf-id", default="Qwen/Qwen3-8B")
    p.add_argument("--target-model", default="olmo-3-7b-think",
                   help="released model whose files carry Young's target letters")
    p.add_argument("--hints", default="sycophancy,grader")
    p.add_argument("--engine", default="auto", choices=["auto", "vllm", "hf", "none"],
                   help="none: generate nothing; restore an earlier session's chains and parse them")
    p.add_argument("--vllm-dir", default="/tmp/s6_venv")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--chunk", type=int, default=96, help="vLLM prompts per write")
    p.add_argument("--hf-batch", type=int, default=8)
    p.add_argument("--hf-max-new", type=int, default=4096)
    p.add_argument("--max-questions", type=int, default=0)
    p.add_argument("--budget-hours", type=float, default=3.5)
    p.add_argument("--skip-deps", action="store_true", help="tests: leave packages alone")
    p.add_argument("--reserve-hours", type=float, default=5.0,
                   help="session time left for the next cell's analyses")
    # worker-only
    p.add_argument("--jobs", default="")
    p.add_argument("--out", default="")
    p.add_argument("--deadline", type=float, default=0.0)
    p.add_argument("--eager", action="store_true")
    return p


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    a.hints = [h for h in a.hints.split(",") if h] if isinstance(a.hints, str) else a.hints
    if a.cmd == "worker":
        cmd_worker(a)
    else:
        cmd_all(a)


# first session: "all --budget-hours 3.0" (960 of 1494 prompts in 3 h).
# resume session: restore those chains from the added input, generate nothing.
NOTEBOOK_ARGV = "all --engine none"


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        return get_ipython() is not None and not _IN_FILE
    except Exception:
        return False


def _cell_source() -> Optional[str]:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        try:
            src = ip.get_parent()["content"]["code"]
        except Exception:
            src = None
        if not src:
            hist = ip.user_ns.get("_ih") or ip.user_ns.get("In") or []
            src = hist[-1] if hist else None
        return src if src and "study6" in src and "def worker_vllm" in src else None
    except Exception:
        return None


if __name__ == "__main__":
    if _in_notebook() and not os.environ.get("S6_CHILD"):
        argv = os.environ.get("S6_ARGS", NOTEBOOK_ARGV).split()
        print(f"notebook run: {argv}", flush=True)
        try:
            write_session_deadline()
            src = _cell_source()
            if src:
                f = Path.cwd() / "study6_qwen_generate.py"
                f.write_text(src, encoding="utf-8")
                pr = subprocess.Popen([sys.executable, "-u", str(f), *argv], stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1,
                                      env=dict(os.environ, S6_CHILD="1"))
                assert pr.stdout is not None
                for line in pr.stdout:
                    print(line, end="", flush=True)
                print(f"study6 generation finished with exit code {pr.wait()}", flush=True)
            else:
                print("cell source not available; running in this kernel", flush=True)
                main(argv)
        except BaseException as e:      # never stop the notebook: the next cell must run
            print(f"study6 stopped: {type(e).__name__}: {e}", flush=True)
    else:
        main()
