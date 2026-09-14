#!/usr/bin/env python3
"""Post-hoc weight averaging (SWA-style) over training snapshots already on disk.

A zero-training proxy for weight-EMA: instead of maintaining an EMA during
training, average the float weights of N periodic snapshots after the fact.
The classic recipe (SWA, Izmailov et al. 2018; "model soup" style averaging)
is a uniform average over the last half of training.

Design notes
------------
* CPU only. Never touches the GPU -- checkpoints are loaded with
  ``map_location="cpu"`` and nothing is ever moved to a device.
* Output format is byte-for-byte the same *shape* as a normal checkpoint written
  by ``common/checkpoint.py``::

      {"step": int, "model": state_dict, "optim": ..., "grad_scaler": ...}

  Only ``model``'s float tensors are averaged. Every other top-level entry
  (optimizer state, step, grad scaler) is copied verbatim from the LAST
  snapshot, so the file loads exactly like a normal checkpoint and
  ``Checkpointer.load`` prints the usual
  "successfully loaded state dict for model" line that ``diffusion/eval.py``
  relies on.
* Non-float entries of the model state dict (integer buffers such as
  ``num_batches_tracked``, bool masks) are NOT averaged -- they are copied from
  the last snapshot.
* An extra top-level key ``averaged_from`` records the provenance (source steps,
  weighting scheme, per-snapshot weights). It holds only plain builtins so the
  file still loads under ``torch.load(..., weights_only=True)``, which is the
  default since torch 2.6. ``Checkpointer.load`` ignores unregistered keys.

Usage
-----
    python scripts/average_checkpoints.py \
        --run logs/diffusion_debug/v1.61-fs.61.fs_p1_X_500k.61 \
        --steps 250000,300000,350000,400000,450000,500000 \
        --out avg_250k_500k_uniform.ckpt \
        --weights uniform

``--steps`` also accepts a range shorthand ``250000:500000:50000``
(start:stop:step, inclusive of stop).
"""

import argparse
import datetime
import os
import sys

import torch

MODEL_KEY = "model"


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #
def load_ckpt(path):
    """Load a checkpoint onto the CPU, tolerating old/new torch.load defaults."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location="cpu")
    except Exception:
        # Checkpoint holds something the weights_only unpickler rejects.
        # These files are produced by this repo's own training loop, so the
        # fallback is safe here.
        return torch.load(path, map_location="cpu", weights_only=False)


def parse_steps(spec):
    """'a,b,c' or 'start:stop:step' -> sorted list of ints."""
    spec = spec.strip()
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"range spec must be start:stop:step, got {spec!r}")
        start, stop, step = (int(p) for p in parts)
        if step <= 0:
            raise ValueError("range step must be positive")
        steps = list(range(start, stop + 1, step))
    else:
        steps = [int(p) for p in spec.replace(" ", "").split(",") if p]
    if not steps:
        raise ValueError("no steps parsed")
    if len(set(steps)) != len(steps):
        raise ValueError(f"duplicate steps in {steps}")
    return sorted(steps)


def make_weights(steps, scheme):
    """Per-snapshot weights, normalised to sum to 1.0.

    uniform : every snapshot counts the same (standard SWA).
    linear  : weight proportional to the snapshot's 1-based position in the
              (sorted) list, so later snapshots count more. With 6 snapshots the
              weights are 1/21 .. 6/21.
    """
    n = len(steps)
    if scheme == "uniform":
        raw = [1.0] * n
    elif scheme == "linear":
        raw = [float(i + 1) for i in range(n)]
    else:
        raise ValueError(f"unknown weighting scheme {scheme!r}")
    total = sum(raw)
    return [w / total for w in raw]


# --------------------------------------------------------------------------- #
# averaging
# --------------------------------------------------------------------------- #
def average_checkpoints(run_dir, steps, scheme, verbose=True):
    """Return (averaged_ckpt_dict, info_dict). Streams snapshots one at a time."""
    paths = []
    for s in steps:
        p = os.path.join(run_dir, f"step_{s}.ckpt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing snapshot: {p}")
        paths.append(p)

    weights = make_weights(steps, scheme)

    # The LAST snapshot is the base: it supplies every non-model entry
    # (optim / grad_scaler / step) and every non-float model entry.
    base = load_ckpt(paths[-1])
    if not isinstance(base, dict) or MODEL_KEY not in base:
        raise ValueError(
            f"{paths[-1]} is not a checkpoint dict with a {MODEL_KEY!r} key "
            f"(top-level type {type(base)})"
        )
    ref_sd = base[MODEL_KEY]
    ref_keys = list(ref_sd.keys())

    float_keys = [
        k for k in ref_keys
        if torch.is_tensor(ref_sd[k]) and ref_sd[k].is_floating_point()
    ]
    copied_keys = [k for k in ref_keys if k not in set(float_keys)]

    if verbose:
        print(f"[avg] run        : {run_dir}")
        print(f"[avg] snapshots  : {steps}")
        print(f"[avg] weighting  : {scheme} -> "
              f"{[round(w, 6) for w in weights]}")
        print(f"[avg] model keys : {len(ref_keys)} "
              f"({len(float_keys)} float tensors averaged, "
              f"{len(copied_keys)} copied from step_{steps[-1]})")
        if copied_keys:
            print(f"[avg] copied     : {copied_keys}")
        print(f"[avg] other top-level entries copied from step_{steps[-1]}: "
              f"{[k for k in base if k != MODEL_KEY]}")

    # float64 accumulator: 6 float32 adds is harmless, but this costs little
    # (model is ~6.3M params) and removes any doubt about accumulation error.
    acc = {k: torch.zeros_like(ref_sd[k], dtype=torch.float64) for k in float_keys}

    for path, step, w in zip(paths, steps, weights):
        ckpt = load_ckpt(path)
        if not isinstance(ckpt, dict) or MODEL_KEY not in ckpt:
            raise ValueError(f"{path} is not a checkpoint dict with {MODEL_KEY!r}")
        sd = ckpt[MODEL_KEY]
        if set(sd.keys()) != set(ref_keys):
            missing = set(ref_keys) - set(sd.keys())
            extra = set(sd.keys()) - set(ref_keys)
            raise ValueError(
                f"state dict key mismatch in {path}: missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )
        for k in float_keys:
            t = sd[k]
            if not torch.is_tensor(t) or t.shape != ref_sd[k].shape:
                raise ValueError(
                    f"shape/type mismatch for {k!r} in {path}: "
                    f"{getattr(t, 'shape', type(t))} vs {ref_sd[k].shape}"
                )
            acc[k].add_(t.to(torch.float64), alpha=w)
        if verbose:
            print(f"[avg]   + step_{step} (weight {w:.6f})")
        del ckpt, sd

    averaged_sd = dict(ref_sd)  # preserves key order; non-float entries kept
    for k in float_keys:
        averaged_sd[k] = acc[k].to(ref_sd[k].dtype)

    out = dict(base)
    out[MODEL_KEY] = averaged_sd
    info = {
        "source_run": os.path.abspath(run_dir),
        "source_steps": list(steps),
        "source_files": [os.path.basename(p) for p in paths],
        "weighting": scheme,
        "weights": [float(w) for w in weights],
        "base_snapshot": os.path.basename(paths[-1]),
        "num_model_entries": len(ref_keys),
        "num_averaged_tensors": len(float_keys),
        "copied_model_entries": list(copied_keys),
        "created_by": "scripts/average_checkpoints.py",
        "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "note": (
            "Float tensors of 'model' are a weighted average of the listed "
            "snapshots. All other entries come from base_snapshot."
        ),
    }
    out["averaged_from"] = info
    return out, info


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Average the model weights of several training snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run", required=True,
                    help="run directory containing step_<N>.ckpt snapshots")
    ap.add_argument("--steps", required=True,
                    help="comma list (250000,300000,...) or start:stop:step")
    ap.add_argument("--out", required=True,
                    help="output filename, written inside --run")
    ap.add_argument("--weights", default="uniform", choices=["uniform", "linear"],
                    help="uniform = plain SWA; linear = later snapshots count more")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the output file if it already exists")
    args = ap.parse_args(argv)

    run_dir = os.path.abspath(os.path.expanduser(args.run))
    if not os.path.isdir(run_dir):
        ap.error(f"run dir not found: {run_dir}")
    if os.path.dirname(args.out):
        ap.error("--out must be a bare filename; it is written inside --run")
    out_path = os.path.join(run_dir, args.out)
    if os.path.exists(out_path) and not args.force:
        ap.error(f"refusing to overwrite existing {out_path} (use --force)")

    steps = parse_steps(args.steps)
    ckpt, info = average_checkpoints(run_dir, steps, args.weights)

    tmp_path = out_path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, out_path)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"[avg] wrote {out_path} ({size_mb:.1f} MiB)")
    print(f"[avg] averaged_from = {info['weighting']} over {info['source_steps']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
