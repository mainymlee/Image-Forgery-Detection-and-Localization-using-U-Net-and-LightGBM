import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

BASE_DIR = "/content/drive/MyDrive/forensic_project"
sys.path.insert(0, BASE_DIR)

import step02_train_stage1 as s2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True, help="0, 1, 2 중 하나")
    args = parser.parse_args()

    if args.run_id not in [0, 1, 2]:
        raise ValueError("run_id는 0, 1, 2 중 하나여야 합니다.")

    seed = s2.CFG["seeds"][args.run_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")
    print(f"Run {args.run_id + 1}/3만 실행합니다. seed={seed}")

    df = pd.read_csv(s2.CSV_PATH)
    train_df = df[df["split"] == "train"].reset_index(drop=True)

    missing_img = (~df["image_path"].map(os.path.exists)).sum()
    missing_msk = (~df["mask_path"].map(os.path.exists)).sum()
    if missing_img or missing_msk:
        raise FileNotFoundError(f"경로 오류: missing images={missing_img}, missing masks={missing_msk}")

    groups = train_df["group_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)

    split_n = int(len(groups) * (1 - s2.CFG["inner_val_ratio"]))
    inner_train_grps = set(groups[:split_n])
    inner_val_grps = set(groups[split_n:])

    inner_train_df = train_df[train_df["group_id"].isin(inner_train_grps)]
    inner_val_df = train_df[train_df["group_id"].isin(inner_val_grps)]

    best_dice, ckpt_path, epoch_log = s2.train_single_run(
        inner_train_df,
        inner_val_df,
        run_id=args.run_id,
        seed=seed,
        device=device
    )

    os.makedirs(f"{BASE_DIR}/results", exist_ok=True)
    log_path = f"{BASE_DIR}/results/stage1_run{args.run_id}_log.json"

    log = {
        "run": args.run_id,
        "seed": seed,
        "best_val_dice": float(best_dice),
        "ckpt_path": ckpt_path,
        "epoch_log": epoch_log
    }

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"Run {args.run_id + 1}/3 완료")
    print(f"best dice: {best_dice}")
    print(f"checkpoint: {ckpt_path}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
