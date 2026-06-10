================================================================
합성 이미지 탐지 프로젝트 v2 — 512 + U-Net++ 개선 실험
================================================================

기존 forensic_project 결과는 보존하고, 이 코드는 forensic_project_v2에 저장합니다.

v2 변경 사항:
  - Stage 1 model: U-Net++ + EfficientNet-B4
  - target_size: 512
  - batch_size: 4  (OOM 발생 시 2로 낮추기)
  - num_workers: 2 (worker 오류 또는 메모리 문제가 있으면 0으로 낮추기)
  - pos_weight: 4.0
  - mask process: JPG probe_mask threshold 후 3x3 morphological closing
  - run: 1회만 실행
  - Stage 2 min_area: 400
  - Stage 2 threshold/top-k 후보는 기존처럼 0.4/0.5/0.6 × 3/5

Drive 업로드 위치:
  /content/drive/MyDrive/forensic_uploads/defacto_15000_probe.zip
  /content/drive/MyDrive/forensic_uploads/coco_real_15000.zip
  /content/drive/MyDrive/forensic_uploads/forensic_project_code_v2_flat.zip

Colab 실행 순서:
  from google.colab import drive
  drive.mount('/content/drive', force_remount=True)

  !pip install -q segmentation-models-pytorch albumentations lightgbm shap scikit-image timm opencv-python-headless joblib

  !rm -rf /content/forensic_project_v2
  !mkdir -p /content/forensic_project_v2
  !unzip -o -q /content/drive/MyDrive/forensic_uploads/forensic_project_code_v2_flat.zip -d /content/forensic_project_v2

  !mkdir -p /content/drive/MyDrive/forensic_project_v2
  !cp /content/forensic_project_v2/*.py /content/drive/MyDrive/forensic_project_v2/
  !cp /content/forensic_project_v2/README.txt /content/drive/MyDrive/forensic_project_v2/ 2>/dev/null || true

  !mkdir -p /content/zips
  !cp /content/drive/MyDrive/forensic_uploads/defacto_15000_probe.zip /content/zips/
  !cp /content/drive/MyDrive/forensic_uploads/coco_real_15000.zip /content/zips/

  !python /content/drive/MyDrive/forensic_project_v2/step01_data_prep.py --clean     --defacto_zip /content/zips/defacto_15000_probe.zip     --coco_zip /content/zips/coco_real_15000.zip

  !python -u /content/drive/MyDrive/forensic_project_v2/step02_train_stage1.py
  !python -u /content/drive/MyDrive/forensic_project_v2/step03_heatmap_features.py
  !python -u /content/drive/MyDrive/forensic_project_v2/step04_train_stage2.py
  !python -u /content/drive/MyDrive/forensic_project_v2/step05_evaluate.py

OOM 대응:
  1) step02_train_stage1.py에서 batch_size 4 -> 2
  2) 그래도 문제가 있으면 num_workers 2 -> 0
  3) 그래도 OOM이면 step02/step03의 UnetPlusPlus를 Unet으로 되돌리기

주의:
  - 기존 forensic_project 폴더를 덮어쓰지 않습니다.
  - 512로 학습했으면 step03/step05도 512 기준이어야 합니다.
================================================================
