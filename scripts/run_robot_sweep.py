#!/usr/bin/env python
"""Hyperparameter sweep for cake.sh (FLUX.1-Kontext on PIE-Bench sample 000000000001).

Sweeps over:
    threshold = bhc_tau_high in [0.70, 0.74, 0.78]
    bhc_lambda_max          in [0.15, 0.20, 0.30, 0.40]
    warmup_steps            in [3, 4, 5]
    alpha                   in [1.0, 1.2, 1.5, 2.0]

Derived:
    bhc_tau_low = threshold - 0.30

Fixed (from cake.sh):
    backend=flux1-kontext, model_path/pie_root as in repo, gpu_id=4, seed=42,
    guidance_scale=2.5, gaussian_sigma=2, hav_steps=10, beta=1,
    sample_ids=000000000025.

Total: 3 x 4 x 3 x 4 = 144 configurations.

Outputs land under
/data_ljy/ll/output/results/PIE_Bench/flux_kontext_dev_havedit_demo/bianli/<name>/
with a top-level sweep_log_<sample>.csv index and per-run metadata.json.

Usage:
    python scripts/run_cake_sweep.py                       # run all
    python scripts/run_cake_sweep.py --list                # print plan only
    python scripts/run_cake_sweep.py --only cake_th074_lam030_wu4_a10
    python scripts/run_cake_sweep.py --start-from cake_th074_lam020_wu3_a10
    python scripts/run_cake_sweep.py --force-all           # ignore existing outputs
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(
    "/data_ljy/ll/output/results/PIE_Bench/flux_kontext_dev_havedit_demo/bianli"
)
SAMPLE_ID = "000000000025"

BASE_ARGS: dict[str, object] = {
    "backend": "flux1-kontext",
    "model_path": "/data_ljy/ll/weight/FLUX.1-Kontext-dev",
    "pie_root": "/data_ljy/ll/dataset/PIE_Bench",
    "gpu_id": 6,
    "seed": 42,
    "guidance_scale": 2.5,
    "gaussian_sigma": 2,
    "hav_steps": 10,
    "beta": 1,
}

THRESHOLDS = [0.70, 0.74, 0.78]
LAMBDA_MAXES = [0.15, 0.20, 0.30, 0.40]
WARMUP_STEPS = [3, 4, 5]
ALPHAS = [1.0, 1.2, 1.5, 2.0]

TAU_LOW_OFFSET = 0.30

SWEPT_KEYS = [
    "threshold",
    "bhc_tau_high",
    "bhc_tau_low",
    "bhc_lambda_max",
    "warmup_steps",
    "alpha",
]


def _fmt2(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def _fmt_alpha(value: float) -> str:
    return f"{int(round(value * 10)):02d}"


def build_configs() -> list[tuple[str, dict[str, object]]]:
    configs: list[tuple[str, dict[str, object]]] = []
    for threshold, lam, warmup, alpha in itertools.product(
        THRESHOLDS, LAMBDA_MAXES, WARMUP_STEPS, ALPHAS
    ):
        tau_low = round(threshold - TAU_LOW_OFFSET, 4)
        overrides: dict[str, object] = {
            "threshold": threshold,
            "bhc_tau_high": threshold,
            "bhc_tau_low": tau_low,
            "bhc_lambda_max": lam,
            "warmup_steps": warmup,
            "alpha": alpha,
        }
        name = (
            f"cake_th{_fmt2(threshold)}"
            f"_lam{_fmt2(lam)}"
            f"_wu{warmup}"
            f"_a{_fmt_alpha(alpha)}"
        )
        configs.append((name, overrides))
    return configs


def kebab(name: str) -> str:
    return name.replace("_", "-")


def cli_args_from(config: dict) -> list[str]:
    args: list[str] = []
    for k, v in config.items():
        args.extend(["--" + kebab(k), str(v)])
    return args


def find_generated_image(run_dir: Path) -> Path | None:
    for cand in run_dir.rglob("strength_1_00.jpg"):
        if cand.parent.name == SAMPLE_ID:
            return cand
    return None


def metadata_path(run_dir: Path) -> Path:
    return run_dir / f"metadata_{SAMPLE_ID}.json"


def cmd_path(run_dir: Path) -> Path:
    return run_dir / f"cmd_{SAMPLE_ID}.txt"


def is_completed(run_dir: Path) -> bool:
    meta = metadata_path(run_dir)
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if data.get("status") != "ok":
        return False
    gen_rel = data.get("generated_image")
    if not gen_rel:
        return False
    gen_path = run_dir.parent / gen_rel
    return gen_path.exists()


def run_one(
    name: str,
    overrides: dict,
    output_root: Path,
    *,
    force: bool,
) -> dict:
    config = {**BASE_ARGS, **overrides}
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)

    if not force and is_completed(run_dir):
        meta = json.loads(metadata_path(run_dir).read_text())
        meta["status"] = "skipped"
        return meta

    cmd: list[str] = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_piebench_batch.py"),
        "--sample-ids", SAMPLE_ID,
        "--output-dir", str(run_dir),
        "--overwrite",
    ]
    cmd.extend(cli_args_from(config))

    cmd_path(run_dir).write_text(" \\\n  ".join(cmd) + "\n")
    metadata = {
        "name": name,
        "overrides": overrides,
        "config": config,
        "swept_keys": SWEPT_KEYS,
        "sample_id": SAMPLE_ID,
        "status": "running",
    }
    metadata_path(run_dir).write_text(json.dumps(metadata, indent=2))

    start = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - start

    metadata["returncode"] = result.returncode
    metadata["elapsed_seconds"] = round(elapsed, 1)
    if result.returncode == 0:
        gen = find_generated_image(run_dir)
        if gen is not None:
            metadata["generated_image"] = str(gen.relative_to(output_root))
            out_png = run_dir / f"{SAMPLE_ID}.png"
            try:
                from PIL import Image

                with Image.open(gen) as img:
                    img.save(out_png, "PNG")
            except Exception:
                shutil.copy2(gen, out_png)
            metadata["flat_image"] = str(out_png.relative_to(output_root))
            metadata["status"] = "ok"
        else:
            metadata["generated_image"] = None
            metadata["status"] = "no_output"
    else:
        metadata["generated_image"] = None
        metadata["status"] = "failed"

    metadata_path(run_dir).write_text(json.dumps(metadata, indent=2))
    return metadata


def write_csv(rows: list[dict], csv_path: Path) -> None:
    columns = [
        "name",
        "status",
        "elapsed_seconds",
        "flat_image",
        "generated_image",
        *SWEPT_KEYS,
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            cfg = row.get("config", {})
            writer.writerow(
                {
                    "name": row.get("name"),
                    "status": row.get("status"),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "flat_image": row.get("flat_image"),
                    "generated_image": row.get("generated_image"),
                    **{k: cfg.get(k) for k in SWEPT_KEYS},
                }
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Sweep root directory.",
    )
    parser.add_argument(
        "--only",
        help="Only run a single named config.",
    )
    parser.add_argument(
        "--start-from",
        help="Skip configs (in plan order) before this name.",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Re-run configs even if a previous successful run is detected.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the planned configs and exit (no runs).",
    )
    return parser.parse_args(argv)


def select_configs(args: argparse.Namespace) -> list[tuple[str, dict]]:
    configs = build_configs()
    if args.only:
        configs = [(n, o) for n, o in configs if n == args.only]
        if not configs:
            raise SystemExit(f"--only {args.only}: no matching config")
    elif args.start_from:
        idx = next(
            (i for i, (n, _) in enumerate(configs) if n == args.start_from),
            None,
        )
        if idx is None:
            raise SystemExit(f"--start-from {args.start_from}: no matching config")
        configs = configs[idx:]
    return configs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    configs = select_configs(args)

    if args.list:
        for name, overrides in configs:
            print(f"{name}: {overrides}")
        print(f"\n({len(configs)} configs)")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    sweep_start = time.time()
    for i, (name, overrides) in enumerate(configs, 1):
        print(
            f"\n===== [{i}/{len(configs)}] {name} =====\n"
            f"overrides: {overrides}"
        )
        row = run_one(name, overrides, output_root, force=args.force_all)
        rows.append(row)
        print(
            f"-> status={row.get('status')} "
            f"elapsed={row.get('elapsed_seconds', '?')}s "
            f"image={row.get('generated_image')}"
        )

    csv_path = output_root / f"sweep_log_{SAMPLE_ID}.csv"
    write_csv(rows, csv_path)

    total_elapsed = time.time() - sweep_start
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_skipped = sum(1 for r in rows if r.get("status") == "skipped")
    n_failed = sum(1 for r in rows if r.get("status") in {"failed", "no_output"})
    print(
        f"\n========== Sweep done in {total_elapsed:.1f}s ==========\n"
        f"ok={n_ok} skipped={n_skipped} failed={n_failed}\n"
        f"CSV index: {csv_path}\n"
        f"Per-run dirs: {output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
