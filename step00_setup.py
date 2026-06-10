"""
Step 00 — 라이브러리 설치 및 Google Drive 마운트
Colab에서 가장 먼저 실행하세요.
"""

import os
import subprocess
import sys

packages = [
    "segmentation-models-pytorch",
    "albumentations",
    "lightgbm",
    "shap",
    "scikit-image",
    "timm",
    "opencv-python-headless",
    "joblib",
]

for pkg in packages:
    print(f"install/check: {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("라이브러리 설치 완료")

# Drive mount는 노트북 셀에서 직접 수행하는 것이 가장 안정적입니다.
# 이미 mount되어 있으면 그대로 진행하고, 스크립트 실행 환경에서 mount가 실패해도 설치는 완료됩니다.
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    print("Google Drive 마운트 완료")
except Exception as e:
    print(f"Drive mount는 건너뜁니다: {e}")
    print("필요하면 노트북 셀에서 직접 실행하세요: from google.colab import drive; drive.mount('/content/drive', force_remount=True)")

BASE_DIR = "/content/drive/MyDrive/forensic_project_v2"
UPLOAD_DIR = "/content/drive/MyDrive/forensic_uploads"

dirs = [
    f"{BASE_DIR}/data",
    f"{BASE_DIR}/checkpoints",
    f"{BASE_DIR}/heatmaps",
    f"{BASE_DIR}/features",
    f"{BASE_DIR}/results/visualizations",
    UPLOAD_DIR,
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

print(f"프로젝트 폴더 준비 완료: {BASE_DIR}")
print(f"데이터 zip 업로드 권장 위치: {UPLOAD_DIR}")
print("필요 zip 파일:")
print("  - defacto_15000_probe.zip")
print("  - coco_real_15000.zip")
