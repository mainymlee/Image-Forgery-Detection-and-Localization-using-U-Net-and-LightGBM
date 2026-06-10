import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

BASE_DIR = "/content/drive/MyDrive/forensic_project"
sys.path.insert(0, BASE_DIR)

from utils import ForensicDataset, compute_stage1_metrics
import step02_train_stage1 as s2


@torch.no_grad()
def evaluate_ckpt(ckpt_path, val_df, device):
    model = s2.build_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    val_ds = ForensicDataset(val_df, mode="val", target_size=s2.CFG["target_size"])
    val_dl = DataLoader(
        val_ds,
        batch_size=s2.CFG["batch_size"],
        shuffle=False,
        num_workers=s2.CFG["num_workers"]
    )

    metrics = {"iou": [], "dice": [], "precision": [], "recall": [], "fp_ratio": []}

    for images, masks, labels in val_dl:
        images = images.to(device)
        preds = torch.sigmoid(model(images)).cpu().numpy()
        gts = masks.numpy()
        labs = labels.numpy()

        for pred, gt, lab in zip(preds, gts, labs):
            m = compute_stage1_metrics(pred[0], gt[0])
            if int(lab) == 1 and gt[0].sum() > 0:
                metrics["iou"].append(m["iou"])
                metrics["dice"].append(m["dice"])
                metrics["precision"].append(m["precision"])
                metrics["recall"].append(m["recall"])
            elif int(lab) == 0 and m["fp_area_ratio"] is not None:
                metrics["fp_ratio"].append(m["fp_area_ratio"])

    def mean(x):
        return float(np.mean(x)) if len(x) > 0 else 0.0

    return {k: mean(v) for k, v in metrics.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = pd.read_csv(f"{BASE_DIR}/data/dataset.csv")
    val_df = df[df["split"] == "validation"].reset_index(drop=True)

    ckpts = sorted(glob.glob(f"{BASE_DIR}/checkpoints/run*_best.pth"))
    if len(ckpts) == 0:
        raise RuntimeError("checkpoint가 없습니다.")

    print("찾은 checkpoint:")
    for c in ckpts:
        print(c)

    results = []
    for ckpt in ckpts:
        print(f"\n평가 중: {ckpt}")
        val_metrics = evaluate_ckpt(ckpt, val_df, device)
        print(val_metrics)
        results.append({
            "ckpt_path": ckpt,
            "val_metrics": val_metrics,
            "external_val_dice": val_metrics["dice"]
        })

    best = max(results, key=lambda x: x["external_val_dice"])

    log = {
        "note": "Checkpoints were trained separately or selected after repeated hold-out training.",
        "available_checkpoints": results,
        "best_ckpt_path": best["ckpt_path"],
        "best_overall_dice": best["external_val_dice"],
        "mean_dice": float(np.mean([r["external_val_dice"] for r in results])),
        "std_dice": float(np.std([r["external_val_dice"] for r in results])),
        "val_metrics": best["val_metrics"]
    }

    out_path = f"{BASE_DIR}/results/stage1_training_log.json"
    os.makedirs(f"{BASE_DIR}/results", exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print("\n최종 선택 checkpoint:", best["ckpt_path"])
    print("stage1_training_log 저장:", out_path)


if __name__ == "__main__":
    main()
