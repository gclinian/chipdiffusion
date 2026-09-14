#!/usr/bin/env python3
"""Experiment ledger — the project's machine-readable record of every run.

Design rules (learned from this project's own failure history):
  * UNCONDITIONAL, not hooked.  `record` is called from inside eval.py and
    train_graph.py themselves, so a run launched by hand in a plain shell at
    2am still lands in the record.  `scan` is the backstop for everything the
    in-process call missed.
  * DERIVED, not authored.  Orphan/uncited detection is a join computed at
    render time.  Nothing here is a field a human must remember to maintain.
  * STDLIB ONLY.  This must run when torch is broken (it currently is).
  * NEVER FATAL.  record() is wrapped by its callers in a bare except; losing
    a ledger row must never lose an experiment.

Usage:
  python3 scripts/ledger.py record <run_dir>   # append one run (idempotent)
  python3 scripts/ledger.py scan               # register everything on disk
  python3 scripts/ledger.py status             # regenerate STATUS.md
"""
import csv, hashlib, json, os, re, subprocess, sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs", "diffusion_debug")
LEDGER_DIR = os.path.join(ROOT, "docs", "ledger")
RUNS = os.path.join(LEDGER_DIR, "runs.jsonl")
RESULTS = os.path.join(LEDGER_DIR, "results")
STATUS = os.path.join(ROOT, "STATUS.md")

CIRCUITS = ["adaptec1", "adaptec2", "adaptec3", "adaptec4",
            "bigblue1", "bigblue2", "bigblue3", "bigblue4"]
CORE7 = [0, 1, 2, 3, 4, 6, 7]          # bigblue2 (idx 5) excluded from the headline
CORE6 = [0, 1, 2, 3, 4, 6]             # CORE7 minus bigblue4: the cheap 3-seed protocol
SCREEN = {"0", "4"}                      # adaptec1/bigblue1: the cheap triage pair, a
                                         # deliberate shape, not a truncated 7-circuit run
PAPER7 = 46.89
BASELINE_PREFIX = "ispd2005-s0.base_cu128_"   # our own runs of the paper checkpoint on the CURRENT stack
SIGMA = 0.654                            # largest measured across-seed std (SVDD, n=3)
PART_RE = re.compile(r"_(part\d+|bb\d+(_g\d+)?|adaptec\d+|bigblue\d+)(?=\.|$)")


# ---------- reading a run directory ----------

def _yaml_get(path, keys):
    """Tiny flat-YAML reader — avoids a PyYAML dependency for 5 scalar keys."""
    out = {}
    try:
        with open(path) as f:
            for line in f:
                if line[:1].isspace() or ":" not in line:
                    continue
                k, _, v = line.partition(":")
                k = k.strip()
                if k in keys:
                    out[k] = v.strip().strip("'\"") or None
    except OSError:
        pass
    return out


def read_run(run_dir):
    """Return a ledger row for one run directory, or None if it is not a run."""
    name = os.path.basename(run_dir.rstrip("/"))
    cfg_path = os.path.join(run_dir, "config.yaml")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    has_ckpt = os.path.exists(os.path.join(run_dir, "latest.ckpt"))
    if not os.path.exists(cfg_path) and not os.path.exists(metrics_path):
        return None

    cfg = _yaml_get(cfg_path, {"task", "method", "seed", "from_checkpoint",
                               "guidance_mode", "family", "train_steps"})
    row = {
        "run_id": name,
        "group": PART_RE.sub("", name),      # part1/part2/single-circuit splits collapse here
        "kind": "eval" if os.path.exists(metrics_path) else "train",
        "task": cfg.get("task"),
        "method": cfg.get("method"),
        "seed": cfg.get("seed"),
        "from_checkpoint": cfg.get("from_checkpoint"),
        "guidance": cfg.get("guidance_mode"),
        "family": cfg.get("family"),
        "has_checkpoint": has_ckpt,
        "per_circuit": {},
        "legality": {},
        "provenance": "disk",
    }

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                for r in csv.DictReader(f):
                    r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
                    idx = r.get("idx")
                    if idx is None or not r.get("hpwl_rescaled"):
                        continue
                    row["per_circuit"][idx] = round(float(r["hpwl_rescaled"]) / 100, 4)
                    if r.get("macro_legality"):
                        row["legality"][idx] = round(float(r["macro_legality"]), 4)
            row["metrics_sha256"] = hashlib.sha256(
                open(metrics_path, "rb").read()).hexdigest()[:16]
        except (OSError, ValueError, KeyError):
            pass
        row["mtime"] = datetime.fromtimestamp(
            os.path.getmtime(metrics_path), timezone.utc).astimezone().isoformat(timespec="seconds")
    elif os.path.exists(cfg_path):
        row["mtime"] = datetime.fromtimestamp(
            os.path.getmtime(cfg_path), timezone.utc).astimezone().isoformat(timespec="seconds")
    return row


# ---------- ledger I/O ----------

def load():
    if not os.path.exists(RUNS):
        return {}
    out = {}
    with open(RUNS) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["run_id"]] = r          # later lines win; the file is append-only
    return out


def save(rows):
    os.makedirs(LEDGER_DIR, exist_ok=True)
    with open(RUNS, "w") as f:
        for k in sorted(rows):
            f.write(json.dumps(rows[k], ensure_ascii=False, sort_keys=True) + "\n")


def archive_metrics(run_dir, run_id):
    """Copy metrics.csv verbatim into git. ~2 KB each; makes every number
    survive the 11 GB gitignored logs/ tree being pruned or overwritten."""
    src = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(src):
        return
    os.makedirs(RESULTS, exist_ok=True)
    dst = os.path.join(RESULTS, run_id + ".csv")
    data = open(src, "rb").read()
    if not os.path.exists(dst) or open(dst, "rb").read() != data:
        open(dst, "wb").write(data)


def record(run_dir):
    row = read_run(run_dir)
    if row is None:
        return None
    rows = load()
    rows[row["run_id"]] = row
    save(rows)
    archive_metrics(run_dir, row["run_id"])
    return row["run_id"]


def scan():
    rows = load()
    added = updated = 0
    for name in sorted(os.listdir(LOGS)) if os.path.isdir(LOGS) else []:
        d = os.path.join(LOGS, name)
        if not os.path.isdir(d):
            continue
        row = read_run(d)
        if row is None:
            continue
        if row["run_id"] not in rows:
            added += 1
        elif rows[row["run_id"]] != row:
            updated += 1
        rows[row["run_id"]] = row
        archive_metrics(d, row["run_id"])
    save(rows)
    return added, updated, len(rows)


# ---------- derived views ----------

def merge_groups(rows):
    """Collapse part1/part2/single-circuit splits into one logical result."""
    groups = {}
    for r in rows.values():
        if r["kind"] != "eval":
            continue
        g = groups.setdefault(r["group"], {
            "group": r["group"], "run_ids": [], "per_circuit": {}, "legality": {},
            "from_checkpoint": r.get("from_checkpoint"), "seed": r.get("seed"),
            "guidance": r.get("guidance"), "mtime": r.get("mtime", ""),
        })
        g["run_ids"].append(r["run_id"])
        g["per_circuit"].update(r["per_circuit"])
        g["legality"].update(r["legality"])
        g["mtime"] = max(g["mtime"], r.get("mtime", ""))
    for g in groups.values():
        have = [i for i in CORE7 if str(i) in g["per_circuit"]]
        g["n_circuits"] = len(have)
        g["complete"] = len(have) == 7
        g["avg7"] = (round(sum(g["per_circuit"][str(i)] for i in have) / 7, 3)
                     if g["complete"] else None)
        # cheap 3-seed protocol: the six cheap circuits, bigblue4 (78% of eval time) held at one seed
        have6 = [i for i in CORE6 if str(i) in g["per_circuit"]]
        g["complete6"] = len(have6) == 6
        g["avg6"] = (round(sum(g["per_circuit"][str(i)] for i in have6) / 6, 3)
                     if g["complete6"] else None)
    return groups


def docs_text():
    out = []
    for base, _, files in os.walk(os.path.join(ROOT, "docs")):
        if os.path.join("docs", "ledger") in base:
            continue
        for fn in files:
            if fn.endswith((".md", ".txt")):
                try:
                    out.append(open(os.path.join(base, fn), errors="ignore").read())
                except OSError:
                    pass
    return "\n".join(out)


def env_fingerprint():
    fp = {}
    try:
        fp["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    except Exception:
        fp["gpu"] = "unknown"
    try:
        fp["git"] = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                   capture_output=True, text=True, timeout=20).stdout.strip()
        fp["dirty"] = bool(subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                                          capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception:
        fp["git"], fp["dirty"] = "unknown", False
    return fp


def current_baseline(groups):
    """Anchor for every comparison: mean avg7 of our own complete runs of the paper
    checkpoint on the current stack (base_cu128_* groups). Never a hard-coded number —
    the previous constant (48.691, a broken March-2026 run) misled the project for five
    months. Returns (mean, n) or (None, 0)."""
    import statistics
    def agg(key, flag):
        vals = [g[key] for g in groups.values() if g[flag] and g["group"].startswith(BASELINE_PREFIX)]
        if not vals:
            return None
        sd = round(statistics.stdev(vals), 3) if len(vals) > 1 else None
        return (round(sum(vals) / len(vals), 3), len(vals), sd)
    return {"avg7": agg("avg7", "complete"), "avg6": agg("avg6", "complete6")}


# ---------- STATUS.md ----------

def render_status():
    rows = load()
    groups = merge_groups(rows)
    text = docs_text()
    fp = env_fingerprint()
    now = datetime.now().astimezone()
    L = []
    w = L.append

    w("# STATUS — 自動產生，不要手改")
    w("")
    w(f"由 `python3 scripts/ledger.py status` 產生於 **{now:%Y-%m-%d %H:%M}**。")
    w(f"來源：`docs/ledger/runs.jsonl`（{len(rows)} runs）+ `docs/ledger/results/`（逐筆 metrics.csv 存檔）。")
    w("")
    w(f"環境指紋：GPU `{fp['gpu']}` ・ git `{fp['git']}`"
      + ("（**working tree 有未 commit 變更**）" if fp["dirty"] else ""))
    w("")

    # --- leaderboard, grouped by seed family ---
    fam = {}
    for g in groups.values():
        if g["complete"]:
            fam.setdefault(PART_RE.sub("", g["group"]).rsplit(".", 1)[0].rsplit("_s3", 1)[0],
                           []).append(g)
    w("## Leaderboard（7-circuit avg HPWL，排除 bigblue2，越低越好）")
    w("")
    cb = current_baseline(groups)
    def fmt(t, label):
        if not t: return f"{label} 未量測"
        m, n, sd = t
        return f"{label} **{m}**（n={n}" + (f", sd {sd}" if sd else "") + "）"
    w(f"參考點：paper 已發表 **{PAPER7}** ・ 我們自己跑 paper checkpoint（`base_cu128_*`，現行 stack）："
      f"{fmt(cb['avg7'], 'avg7')} ・ {fmt(cb['avg6'], 'avg6 [無 bigblue4]')}")
    w("")
    w("avg6 = 六個便宜 circuit（bigblue4 佔 eval 時間 78%，3-seed 協定只在 seed 300 跑它）。"
      "同 stack 的 2σ 雜訊帶 = 2 × avg6 sd；差距小於它的結果不算差距。")
    w("")
    w("這張表只含**磁碟上還有 metrics.csv 的 run**。有些歷史結果（Ablation_10k 44.01、"
      "DDPO v2 44.65、AddLoss v1 等）的原始檔已被 eval 目錄碰撞覆蓋，只存在於報告中 — "
      "那些要看 `docs/all_experiments_summary.csv`。兩張表不一致是預期的，差異本身就是資訊。")
    w("")
    w("| avg7 | avg6 | circuits | run group | seed | checkpoint |")
    w("|-----:|-----:|---------:|-----------|------|------------|")
    rows_ = [g for g in groups.values() if g["complete"] or g["complete6"]]
    for g in sorted(rows_, key=lambda x: (x["avg6"] if x["avg6"] is not None else 99, x["avg7"] or 99)):
        ck = (g["from_checkpoint"] or "-").replace("../public-models/large-v2/large-v2.ckpt", "large-v2 (paper)")
        a7 = f"{g['avg7']:.3f}" if g["avg7"] is not None else "—"
        a6 = f"{g['avg6']:.3f}" if g["avg6"] is not None else "—"
        w(f"| {a7} | {a6} | {g['n_circuits']} | `{g['group']}` | {g['seed'] or '-'} | `{ck}` |")
    w("")

    # --- work queues: the three states that actually go missing ---
    partial, screens = [], 0
    for g in groups.values():
        if g["complete"] or g["n_circuits"] == 0:
            continue
        have = {i for i in g["per_circuit"] if int(i) in CORE7}
        if have <= SCREEN:      # the project's deliberate adaptec1/bigblue1 triage shape
            screens += 1
        else:
            partial.append(g)
    w("## ⚠ 中斷的 eval（跑到一半，既不是 done 也不是 orphan）")
    w("")
    w("單張競爭 GPU 上被 kill / OOM / tmux 掉線的 run。")
    w(f"（另有 {screens} 個只跑 adaptec1/bigblue1 的便宜篩選 run，那是刻意的形狀，不列入。）")
    w("")
    if partial:
        for g in sorted(partial, key=lambda x: -x["n_circuits"]):
            have = ",".join(CIRCUITS[int(i)] for i in sorted(g["per_circuit"], key=int) if int(i) in CORE7)
            w(f"- `{g['group']}` — {g['n_circuits']}/7 ({have})　_{g['mtime'][:10]}_")
    else:
        w("- 無")
    w("")

    # trained checkpoints nobody ever evaluated  (join, not an authored field)
    evaluated = {(r.get("from_checkpoint") or "").split("/")[0] for r in rows.values()}
    orphans = [r for r in rows.values()
               if r["kind"] == "train" and r["has_checkpoint"] and r["run_id"] not in evaluated]
    w("## ⚠ 未收割：訓練完成但從未 eval 的 checkpoint")
    w("")
    if orphans:
        for r in sorted(orphans, key=lambda x: x.get("mtime", ""), reverse=True):
            tag = "　（pilot，可能不需要）" if "pilot" in r["run_id"] or "_test" in r["run_id"] else ""
            w(f"- `{r['run_id']}`　_{(r.get('mtime') or '')[:10]}_{tag}")
    else:
        w("- 無")
    w("")

    # results on disk whose NUMBER appears nowhere in docs/ — the F2 detector.
    # Matching on the value, not the run_id, because docs refer to results by
    # method name ("Run H", "SVDD_layered"), never by directory name.
    def is_cited(g):
        if any(tok in text for tok in (g["group"], *g["run_ids"])):
            return True
        vals = [v for v in (g["avg7"], g["avg6"]) if v is not None]
        return any(f"{v:.{d}f}" in text for v in vals for d in (3, 2))

    uncited = [g for g in groups.values() if (g["complete"] or g["complete6"]) and not is_cited(g)]
    w("## ⚠ 未記錄：有完整結果，但這個數字在 docs/ 裡找不到")
    w("")
    w("以 avg7 數值比對（2/3 位小數）而非目錄名，因為文件裡引用結果用的是方法名不是路徑。")
    w("")
    if uncited:
        for g in sorted(uncited, key=lambda x: x["avg7"]):
            v = g['avg7'] if g['avg7'] is not None else g['avg6']
            w(f"- `{g['group']}` {'avg7' if g['avg7'] is not None else 'avg6'}={v:.3f}　_{g['mtime'][:10]}_")
    else:
        w("- 無")
    w("")

    w("## 判讀規則（同時明文寫在 CLAUDE.md，工具關掉也有效）")
    w("")
    w(f"- 已測得的最大 across-seed σ = **{SIGMA}**（SVDD, n=3）。7-circuit avg 差距小於 "
      f"**{2 * SIGMA:.1f}** 的結果**不得用來關閉任何研究方向**。")
    w("- 任何 n=1 的結果都不能成為禁令。禁令要分 **evidential**（有 Δ、有 seed 數）與 "
      "**prudential**（成本效益判斷）— 只有前者需要過統計門檻，後者要明說是判斷不是證據。")
    w("- 決策 band 不得錨定在 n=1 或原始資料已遺失的基準數字上。")
    w("- 任何方法只能對**同 stack、我們自己跑的 paper checkpoint** 比較（上方 anchor），不能對 46.89"
      "（paper 已發表值）或任何硬編碼舊數字比較。換 GPU/torch = 換 random stream，跨 stack 不可混比。")
    w("")
    return "\n".join(L) + "\n"


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "record":
        rid = record(sys.argv[2])
        print(f"ledger: recorded {rid}" if rid else "ledger: not a run directory")
    elif cmd == "scan":
        a, u, t = scan()
        print(f"ledger: +{a} new, {u} updated, {t} total")
        open(STATUS, "w").write(render_status())
        print(f"ledger: wrote {os.path.relpath(STATUS, ROOT)}")
    elif cmd == "status":
        open(STATUS, "w").write(render_status())
        print(f"ledger: wrote {os.path.relpath(STATUS, ROOT)}")
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
