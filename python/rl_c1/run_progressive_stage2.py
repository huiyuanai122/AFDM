"""
Progressive stage-2 runner for C1 policy integration.

Runs experiments in sequence:
1) imitation_only (reward key selectable)
2) warmstart_reinforce (reward key selectable)
3) optional reward_ber warmstart control

Then aggregates eval json files into one comparison file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run progressive stage-2 policy experiments.")
    p.add_argument("--python_exe", type=str, default=sys.executable)
    p.add_argument("--project_root", type=str, default=".")
    p.add_argument("--data", type=str, default="data/oracle_policy_dataset.mat")
    p.add_argument("--snr_min", type=float, default=14.0)
    p.add_argument("--snr_max", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--imitation_epochs", type=int, default=20)
    p.add_argument("--reinforce_epochs", type=int, default=80)
    p.add_argument("--reward_scale", type=float, default=1000.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--run_reward_ber_control", action="store_true")
    p.add_argument("--build_main_package", action="store_true")
    p.add_argument("--detector_csv", type=str, default="results/ber_results_matlab_tsv1.csv")
    p.add_argument("--out_dir", type=str, default="results/rl_c1_stage2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out_root = (root / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    py = args.python_exe
    train_py = str((root / "python" / "rl_c1" / "train_reinforce.py").resolve())
    eval_py = str((root / "python" / "rl_c1" / "eval_policy.py").resolve())
    main_pkg_py = str((root / "python" / "rl_c1" / "build_main_result_package.py").resolve())
    data_path = str((root / args.data).resolve())

    common = [
        "--data",
        data_path,
        "--snr_min",
        str(args.snr_min),
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
        "--split_mode",
        "sequence",
    ]
    if args.snr_max is not None:
        common += ["--snr_max", str(args.snr_max)]

    # 1) imitation only (reward_mix default via reward key)
    exp1 = out_root / "imitation_only_rewardmix"
    exp1.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            py,
            train_py,
            *common,
            "--reward_key",
            "reward",
            "--imitation_epochs",
            str(args.imitation_epochs),
            "--epochs",
            "0",
            "--output_dir",
            str(exp1),
        ],
        cwd=root,
    )
    run_cmd(
        [
            py,
            eval_py,
            "--data",
            data_path,
            "--checkpoint",
            str(exp1 / "best_reinforce_policy.pt"),
            "--snr_min",
            str(args.snr_min),
            "--reward_key",
            "reward",
            "--device",
            args.device,
            "--save_json",
            str(exp1 / "eval_metrics.json"),
        ]
        + (["--snr_max", str(args.snr_max)] if args.snr_max is not None else []),
        cwd=root,
    )

    # 2) warmstart reinforce (reward_mix)
    exp2 = out_root / "warmstart_rewardmix"
    exp2.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            py,
            train_py,
            *common,
            "--reward_key",
            "reward",
            "--imitation_epochs",
            str(args.imitation_epochs),
            "--epochs",
            str(args.reinforce_epochs),
            "--reward_scale",
            str(args.reward_scale),
            "--output_dir",
            str(exp2),
        ],
        cwd=root,
    )
    run_cmd(
        [
            py,
            eval_py,
            "--data",
            data_path,
            "--checkpoint",
            str(exp2 / "best_reinforce_policy.pt"),
            "--snr_min",
            str(args.snr_min),
            "--reward_key",
            "reward",
            "--device",
            args.device,
            "--save_json",
            str(exp2 / "eval_metrics.json"),
        ]
        + (["--snr_max", str(args.snr_max)] if args.snr_max is not None else []),
        cwd=root,
    )

    summary = {
        "imitation_only_rewardmix": load_json(exp1 / "eval_metrics.json"),
        "warmstart_rewardmix": load_json(exp2 / "eval_metrics.json"),
    }

    # 3) optional control: reward_ber
    if args.run_reward_ber_control:
        exp3 = out_root / "warmstart_rewardber"
        exp3.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                py,
                train_py,
                *common,
                "--reward_key",
                "reward_ber",
                "--imitation_epochs",
                str(args.imitation_epochs),
                "--epochs",
                str(args.reinforce_epochs),
                "--reward_scale",
                str(args.reward_scale),
                "--output_dir",
                str(exp3),
            ],
            cwd=root,
        )
        run_cmd(
            [
                py,
                eval_py,
                "--data",
                data_path,
                "--checkpoint",
                str(exp3 / "best_reinforce_policy.pt"),
                "--snr_min",
                str(args.snr_min),
                "--reward_key",
                "reward_ber",
                "--device",
                args.device,
                "--save_json",
                str(exp3 / "eval_metrics.json"),
            ]
            + (["--snr_max", str(args.snr_max)] if args.snr_max is not None else []),
            cwd=root,
        )
        summary["warmstart_rewardber"] = load_json(exp3 / "eval_metrics.json")

    out_json = out_root / "progressive_comparison.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_json}")

    if args.build_main_package:
        # Prefer reward_ber model if available, otherwise use reward_mix.
        if args.run_reward_ber_control:
            best_ckpt = out_root / "warmstart_rewardber" / "best_reinforce_policy.pt"
            reward_key = "reward_ber"
        else:
            best_ckpt = out_root / "warmstart_rewardmix" / "best_reinforce_policy.pt"
            reward_key = "reward"

        run_cmd(
            [
                py,
                main_pkg_py,
                "--data",
                data_path,
                "--checkpoint",
                str(best_ckpt),
                "--reward_key",
                reward_key,
                "--snr_min",
                str(args.snr_min),
                "--device",
                args.device,
                "--detector_csv",
                str((root / args.detector_csv).resolve()),
                "--output_dir",
                str(out_root / "main_package"),
                "--title_suffix",
                "AFDM Detector + C1 Policy",
            ]
            + (["--snr_max", str(args.snr_max)] if args.snr_max is not None else []),
            cwd=root,
        )


if __name__ == "__main__":
    main()
