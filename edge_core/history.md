  834  echo "Creado: ~/IgnisEdge/scripts/regenerate_flame3_zeromasked.py"
  835  python ~/IgnisEdge/scripts/regenerate_flame3_zeromasked.py
  836  cd ~/IgnisEdge/datasets/yolo_fire_thermal_v2
  837  # Estructura de carpetas
  838  echo "=== ESTRUCTURA ===" && ls -la
  839  # Conteo de archivos por split
  840  echo -e "\n=== CONTEO POR SPLIT ===" && for split in train val test; do     n_img=$(ls $split/images/ 2>/dev/null | wc -l);     n_lbl=$(ls $split/labels/ 2>/dev/null | wc -l);     n_empty=$(find $split/labels/ -empty 2>/dev/null | wc -l);     echo "$split: $n_img images, $n_lbl labels ($n_empty empty)"; done
  841  # Tamaño total
  842  echo -e "\n=== TAMAÑO TOTAL ===" && du -sh .
  843  # Ejemplo de label
  844  echo -e "\n=== EJEMPLO DE LABEL (primer archivo non-empty de train) ===" && for f in train/labels/*.txt; do     if [ -s "$f" ]; then         echo "File: $(basename $f)";         echo "Número de polígonos: $(wc -l < $f)";         echo "Primera línea truncada: $(head -1 $f | cut -c 1-120)...";         break;     fi; done
  845  # Metadata
  846  echo -e "\n=== METADATA ===" && cat metadata.json | python -m json.tool | head -40
  847  # Ver 3 imágenes aleatorias del train set
  848  ls ~/IgnisEdge/datasets/yolo_fire_thermal_v2/train/images/ | shuf -n 3 |     while read f; do xdg-open ~/IgnisEdge/datasets/yolo_fire_thermal_v2/train/images/$f; done
  849  cat > ~/IgnisEdge/scripts/yolov8-seg-p2.yaml << 'EOF'
  850  # YOLOv8-seg with P2 head (small object detection at stride 4)
  851  # Based on Ultralytics yolov8-p2.yaml, Segment head instead of Detect
  852  nc: 1  # fire_region
  853  scales:
  854    # [depth, width, max_channels]
  855    n: [0.33, 0.25, 1024]
  856    s: [0.33, 0.50, 1024]
  857    m: [0.67, 0.75, 768]
  858    l: [1.00, 1.00, 512]
  859    x: [1.00, 1.25, 512]
  860  backbone:
  861    - [-1, 1, Conv, [64, 3, 2]]       # 0-P1/2
  862    - [-1, 1, Conv, [128, 3, 2]]      # 1-P2/4
  863    - [-1, 3, C2f, [128, True]]
  864    - [-1, 1, Conv, [256, 3, 2]]      # 3-P3/8
  865    - [-1, 6, C2f, [256, True]]
  866    - [-1, 1, Conv, [512, 3, 2]]      # 5-P4/16
  867    - [-1, 6, C2f, [512, True]]
  868    - [-1, 1, Conv, [1024, 3, 2]]     # 7-P5/32
  869    - [-1, 3, C2f, [1024, True]]
  870    - [-1, 1, SPPF, [1024, 5]]        # 9
  871  head:
  872    - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  873    - [[-1, 6], 1, Concat, [1]]       # cat P4
  874    - [-1, 3, C2f, [512]]             # 12
  875    - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  876    - [[-1, 4], 1, Concat, [1]]       # cat P3
  877    - [-1, 3, C2f, [256]]             # 15 (P3/8)
  878    - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  879    - [[-1, 2], 1, Concat, [1]]       # cat P2 (NEW)
  880    - [-1, 3, C2f, [128]]             # 18 (P2/4 - small object head)
  881    - [-1, 1, Conv, [128, 3, 2]]
  882    - [[-1, 15], 1, Concat, [1]]
  883    - [-1, 3, C2f, [256]]             # 21 (P3/8 refined)
  884    - [-1, 1, Conv, [256, 3, 2]]
  885    - [[-1, 12], 1, Concat, [1]]
  886    - [-1, 3, C2f, [512]]             # 24 (P4/16 refined)
  887    - [-1, 1, Conv, [512, 3, 2]]
  888    - [[-1, 9], 1, Concat, [1]]
  889    - [-1, 3, C2f, [1024]]            # 27 (P5/32 refined)
  890    - [[18, 21, 24, 27], 1, Segment, [nc, 32, 256]]  # Segment(P2, P3, P4, P5)
  891  EOF
  892  echo "Creado: ~/IgnisEdge/scripts/yolov8-seg-p2.yaml"
  893  cd ~/IgnisEdge && source venv/bin/activate
  894  yolo segment train   data=/home/pablo-silva/IgnisEdge/datasets/yolo_fire_thermal_v2/dataset.yaml   model=/home/pablo-silva/IgnisEdge/scripts/yolov8-seg-p2.yaml   pretrained=yolov8m-seg.pt   scale=m   epochs=150 imgsz=640 batch=8   optimizer=AdamW lr0=0.001   weight_decay=0.0005   patience=30 device=0   hsv_h=0 hsv_s=0 hsv_v=0.1   degrees=15 translate=0.1 scale=0.5   fliplr=0.5 flipud=0.3   mosaic=1.0 mixup=0.1   close_mosaic=10   dropout=0.15   label_smoothing=0.1   project=/home/pablo-silva/IgnisEdge/models   name=thermal_v2_m_p2
  895  cd ~/IgnisEdge && source venv/bin/activate
  896  # Limpiar el intento fallido
  897  rm -rf /home/pablo-silva/IgnisEdge/models/thermal_v2_m_p2
  898  yolo segment train   data=/home/pablo-silva/IgnisEdge/datasets/yolo_fire_thermal_v2/dataset.yaml   model=yolov8m-seg.pt   epochs=150 imgsz=640 batch=8   optimizer=AdamW lr0=0.001   weight_decay=0.0005   patience=30 device=0   hsv_h=0 hsv_s=0 hsv_v=0.1   degrees=15 translate=0.1 scale=0.5   fliplr=0.5 flipud=0.3   mosaic=1.0 mixup=0.1   close_mosaic=10   dropout=0.15   project=/home/pablo-silva/IgnisEdge/models   name=thermal_v2_m
  899  cd ~/IgnisEdge && source venv/bin/activate
  900  rm -rf /home/pablo-silva/IgnisEdge/models/thermal_v2_m_p2
  901  rm -rf /home/pablo-silva/IgnisEdge/models/thermal_v2_m
  902  yolo segment train   data=/home/pablo-silva/IgnisEdge/datasets/yolo_fire_thermal_v2/dataset.yaml   model=yolov8s-seg.pt   epochs=200 imgsz=640 batch=16   optimizer=AdamW lr0=0.0005   weight_decay=0.001   patience=40 device=0   hsv_h=0 hsv_s=0 hsv_v=0.1   degrees=15 translate=0.1 scale=0.5   fliplr=0.5 flipud=0.3   mosaic=1.0 mixup=0.1   close_mosaic=15   dropout=0.15   project=/home/pablo-silva/IgnisEdge/models   name=thermal_v2_s
  903  python p3_camera_test.py
  904  sudo $(which python) p3_camera_test.py
  905  cd camara-termica/p3-ir-camera
  906  sudo $(which python) p3_camera_test.py
  907  pip install pytest
  908  sudo $(which python) p3_viewer.py
  909  cp p3_viewer.py ignis_live.py
  910  tensorboard --logdir /home/pablo-silva/IgnisEdge/models/thermal_v2_s
  911  pip install tensorboard
  912  tensorboard --logdir /home/pablo-silva/IgnisEdge/models/thermal_v2_s
  913  pip install tensorboard --break-system-packages
  914  tensorboard --logdir /home/pablo-silva/IgnisEdge/models/thermal_v2_s
  915  sudo apt update
  916  sudo apt install python3-tensorboard
  917  tensorboard --logdir /home/pablo-silva/IgnisEdge/models/thermal_v2_s
  918  ls -l /home/pablo-silva/IgnisEdge/models/thermal_v2_s
  919  tensorboard --logdir /home/pablo-silva/IgnisEdge/models/
  920  python ignis_live.py
  921  cd ~/IgnisEdge && source venv/bin/activate
  922  # Ver qué hay en el repo del driver
  923  ls /home/pablo-silva/IgnisEdge/camara-termica/p3-ir-camera/
  924  # Buscar el P3 (VID 3474, PID 45a2)
  925  lsusb | grep -i 3474
  926  # Ver si tenemos las udev rules
  927  ls /etc/udev/rules.d/ | grep -i p3
  928  sudo tee /etc/udev/rules.d/99-p3-ir.rules > /dev/null << 'EOF'
  929  SUBSYSTEM=="usb", ATTR{idVendor}=="3474", ATTR{idProduct}=="45c2", MODE="0666"
  930  SUBSYSTEM=="usb", ATTR{idVendor}=="3474", ATTR{idProduct}=="45a2", MODE="0666"
  931  EOF
  932  sudo udevadm control --reload-rules
  933  sudo udevadm trigger
  934  cd /home/pablo-silva/IgnisEdge/camara-termica/p3-ir-camera/
  935  source ~/IgnisEdge/venv/bin/activate
  936  pip install -e .
  937  python -c "from p3_camera import P3Camera, raw_to_celsius; print('Driver OK')"
  938  cat > ~/IgnisEdge/scripts/test_p3_capture.py << 'EOF'
  939  """
  940  Test 1: Capture single frame from P3, apply preprocessing, save QA image.
  941  NO model inference yet - just verifying the pipeline end-to-end.
  942  Output: ~/IgnisEdge/qa_p3_capture/frame_TIMESTAMP_qa.png
  943    Side-by-side: [raw thermal colored | hysteresis mask overlay | YOLO input]
  944  """
  945  import sys
  946  import time
  947  from pathlib import Path
  948  import cv2
  949  import numpy as np
  950  sys.path.insert(0, str(Path(__file__).parent))
  951  from preprocess import (
  952      preprocess_radiometric,
  953      hysteresis_mask,
  954      compute_stats,
  955      SEED_TEMP_C,
  956      EXTEND_TEMP_C,
  957  )
  958  from p3_camera import P3Camera, raw_to_celsius
  959  OUT_DIR = Path.home() / "IgnisEdge/qa_p3_capture"
  960  OUT_DIR.mkdir(exist_ok=True)
  961  def visualize(celsius, processed, stats, title):
  962      """Side-by-side QA: raw | mask | YOLO input."""
  963      h, w = celsius.shape
  964      # Panel 1: thermal with inferno colormap (full dynamic range)
  965      raw_norm = np.clip(
  966          (celsius - celsius.min()) / (celsius.max() - celsius.min() + 1e-6) * 255,
  967          0, 255,
  968      ).astype(np.uint8)
  969      raw_colored = cv2.applyColorMap(raw_norm, cv2.COLORMAP_INFERNO)
  970      # Panel 2: hysteresis mask in red over raw
  971      mask = hysteresis_mask(celsius)
  972      mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
  973      mask_rgb[mask] = [0, 0, 255]
  974      mask_overlay = cv2.addWeighted(raw_colored, 0.5, mask_rgb, 0.5, 0)
  975      # Panel 3: YOLO input
  976      processed_bgr = cv2.cvtColor(processed[:, :, 0], cv2.COLOR_GRAY2BGR)
  977      combined = np.hstack([raw_colored, mask_overlay, processed_bgr])
  978      labels = [
  979          "P3 RAW THERMAL",
  980          f"HYST MASK ({EXTEND_TEMP_C:.0f}-{SEED_TEMP_C:.0f}C)",
  981          "YOLO INPUT",
  982      ]
  983      for i, label in enumerate(labels):
  984          cv2.putText(combined, label, (i * w + 10, 18),
  985                      cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
  986      footer = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
  987      line1 = f"{title}  |  shape={celsius.shape}  range=[{stats['min_c']:.1f}, {stats['max_c']:.1f}]C  mean={stats['mean_c']:.1f}C"
  988      line2 = f"seed_px(>={SEED_TEMP_C:.0f}C)={stats['seed_pixels']}  extend_px(>={EXTEND_TEMP_C:.0f}C)={stats['extend_pixels']}  connected={stats['connected_pixels']}  rejected={stats['rejected_pixels']}"
  989      cv2.putText(footer, line1, (10, 18),
  990                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
  991      cv2.putText(footer, line2, (10, 38),
  992                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
  993      return np.vstack([combined, footer])
  994  def main():
  995      print("=" * 70)
  996      print("IgnisEdge - P3 Capture Pipeline Test")
  997      print("=" * 70)
  998      print("\n[1/4] Connecting to P3 camera...")
  999      camera = P3Camera()
 1000      camera.connect()
 1001      print("      Connected.")
 1002      print("\n[2/4] Initializing camera...")
 1003      camera.init()
 1004      time.sleep(0.5)
 1005      print("\n[3/4] Setting LOW GAIN (range 0-550C, required for fire detection)...")
 1006      camera.set_gain_low()
 1007      time.sleep(0.5)
 1008      camera.start_streaming()
 1009      print("      Streaming started.")
 1010      # Discard first 5 frames (camera stabilization)
 1011      print("\n[4/4] Stabilizing (discarding first 5 frames)...")
 1012      for i in range(5):
 1013          camera.read_frame_both()
 1014          time.sleep(0.04)
 1015      # Capture 3 frames spaced 1 second apart
 1016      print("\nCapturing 3 frames (1s apart)...\n")
 1017      for frame_num in range(1, 4):
 1018          ir_brightness, thermal_raw = camera.read_frame_both()
 1019          celsius = raw_to_celsius(thermal_raw)
 1020          stats = compute_stats(celsius)
 1021          processed = preprocess_radiometric(celsius)
 1022          ts = time.strftime("%H%M%S")
 1023          title = f"frame{frame_num}_{ts}"
 1024          viz = visualize(celsius, processed, stats, title)
 1025          out_path = OUT_DIR / f"{title}_qa.png"
 1026          cv2.imwrite(str(out_path), viz)
 1027          print(f"  Frame {frame_num}: range=[{stats['min_c']:.1f}, {stats['max_c']:.1f}]C  "
 1028                f"connected={stats['connected_pixels']}px  "
 1029                f"saved={out_path.name}")
 1030          if frame_num < 3:
 1031              time.sleep(1.0)
 1032      print("\nStopping stream...")
 1033      camera.stop_streaming()
 1034      camera.disconnect()
 1035      print(f"\nDone. QA images saved to: {OUT_DIR}")
 1036      print("\nTo view:")
 1037      print(f"  xdg-open {OUT_DIR}/$(ls -t {OUT_DIR} | head -1)")
 1038  if __name__ == "__main__":
 1039      main()
 1040  EOF
 1041  echo "Creado: ~/IgnisEdge/scripts/test_p3_capture.py"
 1042  python ~/IgnisEdge/scripts/test_p3_capture.py
 1043  python -c "from p3_camera import P3Camera; help(P3Camera.set_gain_mode)"
 1044  sed -i 's|from p3_camera import P3Camera, raw_to_celsius|from p3_camera import P3Camera, raw_to_celsius, GainMode|' ~/IgnisEdge/scripts/test_p3_capture.py
 1045  sed -i 's|camera.set_gain_mode("low")|camera.set_gain_mode(GainMode.LOW)|' ~/IgnisEdge/scripts/test_p3_capture.py
 1046  # Verificar los cambios
 1047  grep -E "GainMode|set_gain" ~/IgnisEdge/scripts/test_p3_capture.py
 1048  from p3_camera import P3Camera, raw_to_celsius, GainMode
 1049  sed -i 's|camera.set_gain_low()|camera.set_gain_mode(GainMode.LOW)|' ~/IgnisEdge/scripts/test_p3_capture.py
 1050  # Verificar
 1051  grep -E "GainMode|set_gain" ~/IgnisEdge/scripts/test_p3_capture.py
 1052  from p3_camera import P3Camera, raw_to_celsius, GainMode
 1053  cd ~/IgnisEdge && source venv/bin/activate
 1054  sed -i 's|camera.set_gain_low()|camera.set_gain_mode(GainMode.LOW)|' ~/IgnisEdge/scripts/test_p3_capture.py
 1055  grep -E "GainMode|set_gain" ~/IgnisEdge/scripts/test_p3_capture.py
 1056  python ~/IgnisEdge/scripts/test_p3_capture.py
 1057  # Encontrar el bus y device del P3
 1058  lsusb | grep 3474
 1059  # Reset USB del P3 (reemplazar XXX/YYY con los tuyos)
 1060  # Si por ejemplo Bus 003 Device 005 → bus=003, device=005
 1061  sudo usb_modeswitch -v 3474 -p 45a2 --reset-usb
 1062  # Ver si hay procesos python colgados
 1063  ps aux | grep -i python | grep -v grep
 1064  # Si ves algún proceso de p3_camera o test_p3_capture, mátalo:
 1065  # (reemplazar PID con el número real)
 1066  # kill -9 PID
 1067  lsusb | grep 3474
 1068  sleep 5
 1069  python ~/IgnisEdge/scripts/test_p3_capture.py
 1070  cd ~/IgnisEdge/camara-termica/p3-ir-camera
 1071  cd ~/IgnisEdge && python ~/IgnisEdge/scripts/test_p3_capture.py
 1072  # Después de replug
 1073  sleep 5 && python ~/IgnisEdge/scripts/test_p3_capture.py
 1074  xdg-open ~/IgnisEdge/qa_p3_capture/$(ls -t ~/IgnisEdge/qa_p3_capture | head -1)
 1075  python ~/IgnisEdge/scripts/test_p3_capture.py
 1076  cat > ~/IgnisEdge/scripts/test_p3_inference.py << 'EOF'
 1077  """
 1078  Test P3 + IgnisEdge model inference end-to-end.
 1079  Captures 5 frames, runs the v1 model on each, saves QA visualizations.
 1080  """
 1081  import sys
 1082  import time
 1083  from pathlib import Path
 1084  import cv2
 1085  import numpy as np
 1086  sys.path.insert(0, str(Path(__file__).parent))
 1087  from preprocess import (
 1088      preprocess_radiometric,
 1089      hysteresis_mask,
 1090      compute_stats,
 1091      SEED_TEMP_C,
 1092      EXTEND_TEMP_C,
 1093  )
 1094  from p3_camera import P3Camera, raw_to_celsius, GainMode
 1095  from ultralytics import YOLO
 1096  MODEL_PATH = Path.home() / "IgnisEdge/models/production/ignisedge_thermal_v1_best.pt"
 1097  OUT_DIR = Path.home() / "IgnisEdge/qa_p3_inference"
 1098  OUT_DIR.mkdir(exist_ok=True)
 1099  N_FRAMES = 5
 1100  def visualize_with_detection(celsius, processed, detections, stats, title):
 1101      """Side-by-side: raw thermal | YOLO input | model detections overlay."""
 1102      h, w = celsius.shape
 1103      # Panel 1: thermal with inferno colormap
 1104      raw_norm = np.clip(
 1105          (celsius - celsius.min()) / (celsius.max() - celsius.min() + 1e-6) * 255,
 1106          0, 255,
 1107      ).astype(np.uint8)
 1108      raw_colored = cv2.applyColorMap(raw_norm, cv2.COLORMAP_INFERNO)
 1109      # Panel 2: YOLO input (what the model sees)
 1110      yolo_input_bgr = cv2.cvtColor(processed[:, :, 0], cv2.COLOR_GRAY2BGR)
 1111      # Panel 3: detections drawn on raw thermal
 1112      detection_overlay = raw_colored.copy()
 1113      n_det = 0
 1114      if detections is not None and len(detections) > 0:
 1115          result = detections[0]
 1116          if result.masks is not None:
 1117              for i, mask_obj in enumerate(result.masks.data):
 1118                  mask = mask_obj.cpu().numpy()
 1119                  # Resize mask to image size
 1120                  mask = cv2.resize(mask, (w, h)) > 0.5
 1121                  # Draw cyan overlay
 1122                  overlay = np.zeros_like(detection_overlay)
 1123                  overlay[mask] = [255, 255, 0]  # BGR cyan
 1124                  detection_overlay = cv2.addWeighted(
 1125                      detection_overlay, 1.0, overlay, 0.4, 0
 1126                  )
 1127                  conf = float(result.boxes.conf[i]) if result.boxes is not None else 0.0
 1128                  n_det += 1
 1129              # Draw bounding boxes
 1130              if result.boxes is not None:
 1131                  for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
 1132                      x1, y1, x2, y2 = map(int, box.cpu().numpy())
 1133                      cv2.rectangle(detection_overlay, (x1, y1), (x2, y2),
 1134                                    (0, 255, 0), 2)
 1135                      cv2.putText(detection_overlay, f"fire {conf:.2f}",
 1136                                  (x1, max(y1 - 5, 12)),
 1137                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
 1138      combined = np.hstack([raw_colored, yolo_input_bgr, detection_overlay])
 1139      labels = ["P3 RAW THERMAL", "YOLO INPUT (zero-masked)", f"DETECTIONS ({n_det})"]
 1140      for i, label in enumerate(labels):
 1141          cv2.putText(combined, label, (i * w + 10, 18),
 1142                      cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
 1143      footer = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
 1144      line1 = f"{title}  |  range=[{stats['min_c']:.1f}, {stats['max_c']:.1f}]C  mean={stats['mean_c']:.1f}C  connected={stats['connected_pixels']}px"
 1145      line2 = f"detections={n_det}  model={MODEL_PATH.name}"
 1146      cv2.putText(footer, line1, (10, 18),
 1147                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
 1148      cv2.putText(footer, line2, (10, 38),
 1149                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
 1150      return np.vstack([combined, footer])
 1151  def main():
 1152      print("=" * 70)
 1153      print("IgnisEdge - P3 + Model Inference Test")
 1154      print("=" * 70)
 1155      if not MODEL_PATH.exists():
 1156          print(f"ERROR: Model not found at {MODEL_PATH}")
 1157          sys.exit(1)
 1158      print(f"\nLoading model: {MODEL_PATH.name}")
 1159      model = YOLO(str(MODEL_PATH))
 1160      print(f"  Classes: {model.names}")
 1161      camera = P3Camera()
 1162      try:
 1163          print("\n[1/4] Connecting...")
 1164          camera.connect()
 1165          print("[2/4] Initializing...")
 1166          camera.init()
 1167          time.sleep(0.5)
 1168          print("[3/4] Setting LOW GAIN...")
 1169          camera.set_gain_mode(GainMode.LOW)
 1170          time.sleep(0.5)
 1171          camera.start_streaming()
 1172          print("      Streaming.")
 1173          print("[4/4] Stabilizing...")
 1174          for _ in range(5):
 1175              camera.read_frame_both()
 1176              time.sleep(0.04)
 1177          print(f"\nCapturing {N_FRAMES} frames + running inference...\n")
 1178          for frame_num in range(1, N_FRAMES + 1):
 1179              ir_brightness, thermal_raw = camera.read_frame_both()
 1180              celsius = raw_to_celsius(thermal_raw)
 1181              stats = compute_stats(celsius)
 1182              processed = preprocess_radiometric(celsius)
 1183              # Run inference
 1184              detections = model.predict(
 1185                  processed,
 1186                  conf=0.25,
 1187                  iou=0.45,
 1188                  verbose=False,
 1189              )
 1190              n_det = 0
 1191              if detections and detections[0].boxes is not None:
 1192                  n_det = len(detections[0].boxes)
 1193              ts = time.strftime("%H%M%S")
 1194              title = f"frame{frame_num}_{ts}"
 1195              viz = visualize_with_detection(celsius, processed, detections, stats, title)
 1196              out_path = OUT_DIR / f"{title}.png"
 1197              cv2.imwrite(str(out_path), viz)
 1198              print(f"  Frame {frame_num}: max={stats['max_c']:.1f}C  "
 1199                    f"connected={stats['connected_pixels']}px  "
 1200                    f"detections={n_det}  saved={out_path.name}")
 1201              if frame_num < N_FRAMES:
 1202                  time.sleep(0.8)
 1203      finally:
 1204          print("\nCleaning up...")
 1205          try:
 1206              camera.stop_streaming()
 1207          except Exception as e:
 1208              print(f"  stop_streaming: {e}")
 1209          try:
 1210              camera.disconnect()
 1211          except Exception as e:
 1212              print(f"  disconnect: {e}")
 1213      print(f"\nDone. Results in: {OUT_DIR}")
 1214  if __name__ == "__main__":
 1215      main()
 1216  EOF
 1217  echo "Creado: ~/IgnisEdge/scripts/test_p3_inference.py"
 1218  sleep 5 && python ~/IgnisEdge/scripts/test_p3_inference.py
 1219  xdg-open ~/IgnisEdge/qa_p3_inference/frame5_213322.png
 1220  SI area_px < 30:                              → DESCARTAR (ruido)
 1221  SI area_px < 200 Y circularity > 0.75:        → POINT_SOURCE (encendedor, motor)
 1222  SI aspect_ratio > 4 Y solidity > 0.85:        → EXTENDED_HOT (escape, chimenea)
 1223  SI temp_gradient < 30°C Y core_ratio > 0.7:   → POINT_SOURCE (calor uniforme = fuente artificial)
 1224  SI halo_extent > 1.8 Y n_subregions >= 2:     → WILDFIRE (fuego forestal con propagación)
 1225  SI halo_extent > 1.5 Y temp_gradient > 80°C:  → WILDFIRE
 1226  EN OTRO CASO:                                  → AMBIGUOUS (revisión humana)
 1227  cd ~/IgnisEdge && source venv/bin/activate
 1228  cat > ~/IgnisEdge/scripts/morphological_classifier.py << 'EOF'
 1229  """
 1230  IgnisEdge - Morphological Thermal Threat Classifier
 1231  Classifies thermal regions from radiometric data using interpretable
 1232  geometric and physical features. No deep learning - all rules are
 1233  explicit and defendable.
 1234  References:
 1235  - Hopkins et al. 2024 (FLAME 3): >200°C threshold for active fire
 1236  - NWCG: canopy heating signatures, halo characteristics
 1237  - Combustion science: pyrolysis 230°C, smoldering 380°C, ignition 590°C
 1238  """
 1239  from __future__ import annotations
 1240  from dataclasses import dataclass, asdict
 1241  from enum import Enum
 1242  from typing import Optional
 1243  import cv2
 1244  import numpy as np
 1245  from scipy.ndimage import label as ndi_label, binary_propagation
 1246  # --- Physical thresholds ---
 1247  EXTEND_TEMP_C = 80.0       # Halo / canopy heating boundary
 1248  SEED_TEMP_C = 150.0        # Pyrolysis / definite hot
 1249  CORE_TEMP_C = 250.0        # Active combustion zone
 1250  INTENSE_TEMP_C = 400.0     # High-intensity fire
 1251  # --- Morphological thresholds (tunable, justified) ---
 1252  MIN_REGION_AREA_PX = 30
 1253  POINT_SOURCE_MAX_AREA = 200
 1254  POINT_SOURCE_MIN_CIRCULARITY = 0.75
 1255  EXTENDED_MIN_ASPECT = 4.0
 1256  EXTENDED_MIN_SOLIDITY = 0.85
 1257  UNIFORM_MAX_GRADIENT_C = 30.0
 1258  HIGH_CORE_RATIO = 0.7
 1259  WILDFIRE_HALO_EXTENT = 1.5
 1260  WILDFIRE_HIGH_HALO = 1.8
 1261  WILDFIRE_HIGH_GRADIENT_C = 80.0
 1262  class ThreatClass(Enum):
 1263      WILDFIRE = "wildfire"
 1264      EXTENDED_HOT = "extended_hot"
 1265      POINT_SOURCE = "point_source"
 1266      AMBIGUOUS = "ambiguous"
 1267      NOISE = "noise"
 1268  class AlertLevel(Enum):
 1269      RED = "red"          # Wildfire + intense
 1270      ORANGE = "orange"    # Wildfire moderate
 1271      YELLOW = "yellow"    # Extended hot (industrial)
 1272      WHITE = "white"      # Ambiguous - human review
 1273      NONE = "none"        # Discarded
 1274  @dataclass
 1275  class RegionFeatures:
 1276      """Geometric + thermal features of one thermal region."""
 1277      region_id: int
 1278      area_px: int
 1279      bbox: tuple  # (x, y, w, h)
 1280      centroid: tuple  # (cx, cy)
 1281      # Thermal
 1282      max_temp_c: float
 1283      mean_temp_c: float
 1284      temp_gradient_c: float  # std of temperatures inside region
 1285      # Geometric
 1286      perimeter: float
 1287      circularity: float
 1288      aspect_ratio: float
 1289      solidity: float
 1290      # Physical
 1291      core_ratio: float        # core_pixels / area
 1292      halo_extent: float       # halo_radius / core_radius
 1293      n_subregions: int        # connected components above CORE_TEMP_C
 1294  @dataclass
 1295  class Detection:
 1296      features: RegionFeatures
 1297      threat_class: ThreatClass
 1298      alert_level: AlertLevel
 1299      reasoning: str
 1300  def _compute_circularity(area: float, perimeter: float) -> float:
 1301      if perimeter <= 0:
 1302          return 0.0
 1303      return float(4.0 * np.pi * area / (perimeter ** 2))
 1304  def _compute_solidity(contour) -> float:
 1305      area = cv2.contourArea(contour)
 1306      hull = cv2.convexHull(contour)
 1307      hull_area = cv2.contourArea(hull)
 1308      if hull_area <= 0:
 1309          return 0.0
 1310      return float(area / hull_area)
 1311  def _equivalent_radius(area_px: float) -> float:
 1312      return float(np.sqrt(area_px / np.pi))
 1313  def extract_region_features(
 1314      celsius: np.ndarray,
 1315      region_mask: np.ndarray,
 1316      region_id: int = 0,
 1317  ) -> Optional[RegionFeatures]:
 1318      """Extract all features for a single connected region."""
 1319      if region_mask.sum() < MIN_REGION_AREA_PX:
 1320          return None
 1321      # Contour analysis
 1322      mask_u8 = region_mask.astype(np.uint8) * 255
 1323      contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
 1324      if not contours:
 1325          return None
 1326      contour = max(contours, key=cv2.contourArea)
 1327      area = cv2.contourArea(contour)
 1328      perimeter = cv2.arcLength(contour, closed=True)
 1329      x, y, w, h = cv2.boundingRect(contour)
 1330      M = cv2.moments(contour)
 1331      if M["m00"] > 0:
 1332          cx = M["m10"] / M["m00"]
 1333          cy = M["m01"] / M["m00"]
 1334      else:
 1335          cx, cy = float(x + w / 2), float(y + h / 2)
 1336      # Thermal stats inside region
 1337      temps_in_region = celsius[region_mask]
 1338      if temps_in_region.size == 0:
 1339          return None
 1340      max_t = float(temps_in_region.max())
 1341      mean_t = float(temps_in_region.mean())
 1342      grad_t = float(temps_in_region.std())
 1343      # Core analysis
 1344      core_mask = region_mask & (celsius >= CORE_TEMP_C)
 1345      core_area = int(core_mask.sum())
 1346      core_ratio = core_area / max(area, 1)
 1347      core_radius = _equivalent_radius(core_area) if core_area > 0 else 0.0
 1348      # Halo extent
 1349      halo_radius = _equivalent_radius(area)
 1350      halo_extent = halo_radius / max(core_radius, 1.0) if core_radius > 0 else 0.0
 1351      # Sub-regions (multiple combustion cores)
 1352      if core_area > 0:
 1353          _, n_sub = ndi_label(core_mask)
 1354      else:
 1355          n_sub = 0
 1356      # Geometry
 1357      aspect = max(w, h) / max(min(w, h), 1)
 1358      circ = _compute_circularity(area, perimeter)
 1359      solid = _compute_solidity(contour)
 1360      return RegionFeatures(
 1361          region_id=region_id,
 1362          area_px=int(area),
 1363          bbox=(int(x), int(y), int(w), int(h)),
 1364          centroid=(float(cx), float(cy)),
 1365          max_temp_c=max_t,
 1366          mean_temp_c=mean_t,
 1367          temp_gradient_c=grad_t,
 1368          perimeter=float(perimeter),
 1369          circularity=circ,
 1370          aspect_ratio=float(aspect),
 1371          solidity=solid,
 1372          core_ratio=float(core_ratio),
 1373          halo_extent=float(halo_extent),
 1374          n_subregions=int(n_sub),
 1375      )
 1376  def classify(f: RegionFeatures) -> tuple[ThreatClass, AlertLevel, str]:
 1377      """Apply morphological rules to classify the region. Returns class + alert + reasoning."""
 1378      reasons = []
 1379      # Rule 0: noise filter
 1380      if f.area_px < MIN_REGION_AREA_PX:
 1381          return ThreatClass.NOISE, AlertLevel.NONE, f"Area {f.area_px}px < {MIN_REGION_AREA_PX} (noise)"
 1382      # Rule 1: small + circular = point source
 1383      if f.area_px < POINT_SOURCE_MAX_AREA and f.circularity > POINT_SOURCE_MIN_CIRCULARITY:
 1384          reasons.append(f"area={f.area_px}<{POINT_SOURCE_MAX_AREA}")
 1385          reasons.append(f"circ={f.circularity:.2f}>{POINT_SOURCE_MIN_CIRCULARITY}")
 1386          return ThreatClass.POINT_SOURCE, AlertLevel.NONE, "Small circular region: " + ", ".join(reasons)
 1387      # Rule 2: elongated + solid = extended hot (industrial)
 1388      if f.aspect_ratio > EXTENDED_MIN_ASPECT and f.solidity > EXTENDED_MIN_SOLIDITY:
 1389          reasons.append(f"aspect={f.aspect_ratio:.1f}>{EXTENDED_MIN_ASPECT}")
 1390          reasons.append(f"solidity={f.solidity:.2f}>{EXTENDED_MIN_SOLIDITY}")
 1391          return ThreatClass.EXTENDED_HOT, AlertLevel.YELLOW, "Elongated solid: " + ", ".join(reasons)
 1392      # Rule 3: uniform temp + high core ratio = artificial (point source)
 1393      if f.temp_gradient_c < UNIFORM_MAX_GRADIENT_C and f.core_ratio > HIGH_CORE_RATIO:
 1394          reasons.append(f"grad={f.temp_gradient_c:.1f}<{UNIFORM_MAX_GRADIENT_C}")
 1395          reasons.append(f"core_ratio={f.core_ratio:.2f}>{HIGH_CORE_RATIO}")
 1396          return ThreatClass.POINT_SOURCE, AlertLevel.NONE, "Uniform high core: " + ", ".join(reasons)
 1397      # Rule 4: large halo + multiple sub-cores = wildfire (strong evidence)
 1398      if f.halo_extent > WILDFIRE_HIGH_HALO and f.n_subregions >= 2:
 1399          level = AlertLevel.RED if f.max_temp_c > INTENSE_TEMP_C else AlertLevel.ORANGE
 1400          reasons.append(f"halo_ext={f.halo_extent:.2f}>{WILDFIRE_HIGH_HALO}")
 1401          reasons.append(f"n_sub={f.n_subregions}>=2")
 1402          reasons.append(f"max_t={f.max_temp_c:.0f}C")
 1403          return ThreatClass.WILDFIRE, level, "Strong wildfire pattern: " + ", ".join(reasons)
 1404      # Rule 5: moderate halo + high gradient = wildfire (moderate evidence)
 1405      if f.halo_extent > WILDFIRE_HALO_EXTENT and f.temp_gradient_c > WILDFIRE_HIGH_GRADIENT_C:
 1406          level = AlertLevel.RED if f.max_temp_c > INTENSE_TEMP_C else AlertLevel.ORANGE
 1407          reasons.append(f"halo_ext={f.halo_extent:.2f}>{WILDFIRE_HALO_EXTENT}")
 1408          reasons.append(f"grad={f.temp_gradient_c:.1f}>{WILDFIRE_HIGH_GRADIENT_C}")
 1409          return ThreatClass.WILDFIRE, level, "Wildfire pattern: " + ", ".join(reasons)
 1410      # Default: ambiguous, needs review
 1411      return ThreatClass.AMBIGUOUS, AlertLevel.WHITE, (
 1412          f"No rule matched: area={f.area_px} circ={f.circularity:.2f} "
 1413          f"aspect={f.aspect_ratio:.1f} grad={f.temp_gradient_c:.1f} "
 1414          f"halo_ext={f.halo_extent:.2f} n_sub={f.n_subregions}"
 1415      )
 1416  def detect_and_classify(celsius: np.ndarray) -> list[Detection]:
 1417      """Full pipeline: hysteresis -> region segmentation -> feature extraction -> classification."""
 1418      # Hysteresis mask (same as preprocess.py)
 1419      seed = celsius >= SEED_TEMP_C
 1420      extend = celsius >= EXTEND_TEMP_C
 1421      if not seed.any():
 1422          return []
 1423      full_mask = binary_propagation(seed, mask=extend)
 1424      # Connected component labeling
 1425      labeled, n_regions = ndi_label(full_mask)
 1426      if n_regions == 0:
 1427          return []
 1428      detections = []
 1429      for region_id in range(1, n_regions + 1):
 1430          region_mask = labeled == region_id
 1431          feats = extract_region_features(celsius, region_mask, region_id)
 1432          if feats is None:
 1433              continue
 1434          threat, alert, reason = classify(feats)
 1435          detections.append(Detection(features=feats, threat_class=threat,
 1436                                      alert_level=alert, reasoning=reason))
 1437      return detections
 1438  if __name__ == "__main__":
 1439      # Self-test
 1440      print("Self-test: synthetic point source vs synthetic wildfire")
 1441      # Synthetic POINT SOURCE: small circle, uniform high temp
 1442      img1 = np.full((200, 200), 25.0, dtype=np.float32)
 1443      cv2.circle(img1, (100, 100), 8, 320.0, -1)
 1444      cv2.circle(img1, (100, 100), 5, 350.0, -1)
 1445      dets1 = detect_and_classify(img1)
 1446      print(f"\nPoint source test: {len(dets1)} regions detected")
 1447      for d in dets1:
 1448          print(f"  -> {d.threat_class.value}  alert={d.alert_level.value}")
 1449          print(f"     {d.reasoning}")
 1450      # Synthetic WILDFIRE: irregular blob with multiple cores and gradient
 1451      img2 = np.full((300, 300), 25.0, dtype=np.float32)
 1452      # Halo
 1453      cv2.ellipse(img2, (150, 150), (80, 60), 30, 0, 360, 100.0, -1)
 1454      cv2.ellipse(img2, (150, 150), (60, 45), 30, 0, 360, 180.0, -1)
 1455      # Multiple cores
 1456      cv2.circle(img2, (130, 145), 12, 380.0, -1)
 1457      cv2.circle(img2, (170, 160), 10, 420.0, -1)
 1458      cv2.circle(img2, (155, 130), 8, 350.0, -1)
 1459      dets2 = detect_and_classify(img2)
 1460      print(f"\nWildfire test: {len(dets2)} regions detected")
 1461      for d in dets2:
 1462          print(f"  -> {d.threat_class.value}  alert={d.alert_level.value}")
 1463          print(f"     {d.reasoning}")
 1464          print(f"     features: area={d.features.area_px}px max={d.features.max_temp_c:.0f}C "
 1465                f"halo_ext={d.features.halo_extent:.2f} n_sub={d.features.n_subregions}")
 1466  EOF
 1467  echo "Creado: ~/IgnisEdge/scripts/morphological_classifier.py"
 1468  python ~/IgnisEdge/scripts/morphological_classifier.py
 1469  cat > ~/IgnisEdge/scripts/test_p3_classifier.py << 'EOF'
 1470  """
 1471  P3 + Morphological Classifier end-to-end test.
 1472  Captures frames, classifies thermal regions, saves QA visualizations
 1473  with alert level and reasoning per region.
 1474  """
 1475  import sys
 1476  import time
 1477  from pathlib import Path
 1478  import cv2
 1479  import numpy as np
 1480  sys.path.insert(0, str(Path(__file__).parent))
 1481  from morphological_classifier import (
 1482      detect_and_classify,
 1483      AlertLevel,
 1484      ThreatClass,
 1485      SEED_TEMP_C,
 1486      EXTEND_TEMP_C,
 1487  )
 1488  from p3_camera import P3Camera, raw_to_celsius, GainMode
 1489  OUT_DIR = Path.home() / "IgnisEdge/qa_p3_classifier"
 1490  OUT_DIR.mkdir(exist_ok=True)
 1491  N_FRAMES = 5
 1492  # Color per alert level (BGR)
 1493  ALERT_COLORS = {
 1494      AlertLevel.RED: (0, 0, 255),
 1495      AlertLevel.ORANGE: (0, 140, 255),
 1496      AlertLevel.YELLOW: (0, 255, 255),
 1497      AlertLevel.WHITE: (255, 255, 255),
 1498      AlertLevel.NONE: (128, 128, 128),
 1499  }
 1500  ALERT_LABEL = {
 1501      AlertLevel.RED: "ROJA",
 1502      AlertLevel.ORANGE: "NARANJA",
 1503      AlertLevel.YELLOW: "AMARILLA",
 1504      AlertLevel.WHITE: "REVISAR",
 1505      AlertLevel.NONE: "DESCARTE",
 1506  }
 1507  def visualize(celsius, detections, title):
 1508      """Side-by-side: raw thermal | classifications overlay | text report."""
 1509      h, w = celsius.shape
 1510      # Panel 1: thermal inferno
 1511      raw_norm = np.clip(
 1512          (celsius - celsius.min()) / (celsius.max() - celsius.min() + 1e-6) * 255,
 1513          0, 255,
 1514      ).astype(np.uint8)
 1515      raw_colored = cv2.applyColorMap(raw_norm, cv2.COLORMAP_INFERNO)
 1516      # Panel 2: detections drawn
 1517      overlay = raw_colored.copy()
 1518      for det in detections:
 1519          x, y, bw, bh = det.features.bbox
 1520          color = ALERT_COLORS[det.alert_level]
 1521          cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)
 1522          label = f"{det.threat_class.value} [{ALERT_LABEL[det.alert_level]}]"
 1523          cv2.putText(overlay, label, (x, max(y - 5, 12)),
 1524                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
 1525      # Panel 3: text report
 1526      report = np.zeros((h, w, 3), dtype=np.uint8)
 1527      cv2.putText(report, "REPORTE", (10, 18),
 1528                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
 1529      if not detections:
 1530          cv2.putText(report, "Sin regiones", (10, 50),
 1531                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
 1532      else:
 1533          y_pos = 40
 1534          for i, det in enumerate(detections):
 1535              color = ALERT_COLORS[det.alert_level]
 1536              cv2.putText(report, f"#{i+1} {det.threat_class.value}",
 1537                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
 1538              y_pos += 14
 1539              cv2.putText(report, f"  alert: {ALERT_LABEL[det.alert_level]}",
 1540                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
 1541              y_pos += 12
 1542              cv2.putText(report, f"  area={det.features.area_px}px",
 1543                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
 1544              y_pos += 11
 1545              cv2.putText(report, f"  max={det.features.max_temp_c:.0f}C grad={det.features.temp_gradient_c:.0f}",
 1546                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
 1547              y_pos += 11
 1548              cv2.putText(report, f"  circ={det.features.circularity:.2f} halo={det.features.halo_extent:.2f}",
 1549                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
 1550              y_pos += 11
 1551              cv2.putText(report, f"  asp={det.features.aspect_ratio:.1f} sub={det.features.n_subregions}",
 1552                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
 1553              y_pos += 18
 1554              if y_pos > h - 20:
 1555                  break
 1556      combined = np.hstack([raw_colored, overlay, report])
 1557      labels_top = ["P3 RAW THERMAL", "CLASSIFICATION", "REPORT"]
 1558      for i, label in enumerate(labels_top):
 1559          cv2.putText(combined, label, (i * w + 10, 18),
 1560                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
 1561      footer = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
 1562      summary = f"{title}  |  range=[{celsius.min():.1f}, {celsius.max():.1f}]C  regions={len(detections)}"
 1563      cv2.putText(footer, summary, (10, 25),
 1564                  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
 1565      return np.vstack([combined, footer])
 1566  def main():
 1567      print("=" * 70)
 1568      print("IgnisEdge - P3 + Morphological Classifier Test")
 1569      print("=" * 70)
 1570      camera = P3Camera()
 1571      try:
 1572          print("\nConnecting...")
 1573          camera.connect()
 1574          camera.init()
 1575          time.sleep(0.5)
 1576          camera.set_gain_mode(GainMode.LOW)
 1577          time.sleep(0.5)
 1578          camera.start_streaming()
 1579          print("Stabilizing...")
 1580          for _ in range(5):
 1581              camera.read_frame_both()
 1582              time.sleep(0.04)
 1583          print(f"\nCapturing {N_FRAMES} frames + classifying...\n")
 1584          for frame_num in range(1, N_FRAMES + 1):
 1585              _, thermal_raw = camera.read_frame_both()
 1586              celsius = raw_to_celsius(thermal_raw)
 1587              detections = detect_and_classify(celsius)
 1588              ts = time.strftime("%H%M%S")
 1589              title = f"frame{frame_num}_{ts}"
 1590              viz = visualize(celsius, detections, title)
 1591              out_path = OUT_DIR / f"{title}.png"
 1592              cv2.imwrite(str(out_path), viz)
 1593              # Print summary
 1594              summary = f"  Frame {frame_num}: max={celsius.max():.1f}C  regions={len(detections)}"
 1595              for det in detections:
 1596                  summary += f"\n    -> {det.threat_class.value} [{ALERT_LABEL[det.alert_level]}]"
 1597                  summary += f"  ({det.reasoning})"
 1598              print(summary)
 1599              print(f"    saved={out_path.name}")
 1600              if frame_num < N_FRAMES:
 1601                  time.sleep(0.8)
 1602      finally:
 1603          print("\nCleaning up...")
 1604          try:
 1605              camera.stop_streaming()
 1606          except Exception as e:
 1607              print(f"  stop_streaming: {e}")
 1608          try:
 1609              camera.disconnect()
 1610          except Exception as e:
 1611              print(f"  disconnect: {e}")
 1612      print(f"\nResults in: {OUT_DIR}")
 1613  if __name__ == "__main__":
 1614      main()
 1615  EOF
 1616  echo "Creado: ~/IgnisEdge/scripts/test_p3_classifier.py"
 1617  sleep 5 && python ~/IgnisEdge/scripts/test_p3_classifier.py
 1618  cat > ~/IgnisEdge/scripts/reset_p3.sh << 'EOF'
 1619  #!/bin/bash
 1620  # Reset P3 camera via /sys USB unbind/bind (no need to unplug cable)
 1621  VID="3474"
 1622  PID="45a2"
 1623  DEVICE=$(for d in /sys/bus/usb/devices/*/; do
 1624      if [ -f "$d/idVendor" ] && [ -f "$d/idProduct" ]; then
 1625          if [ "$(cat $d/idVendor 2>/dev/null)" = "$VID" ] && [ "$(cat $d/idProduct 2>/dev/null)" = "$PID" ]; then
 1626              basename "$d"
 1627              break
 1628          fi
 1629      fi
 1630  done)
 1631  if [ -z "$DEVICE" ]; then
 1632      echo "P3 camera NOT found in USB devices."
 1633      echo "Try: lsusb | grep 3474"
 1634      exit 1
 1635  fi
 1636  echo "Found P3 at: $DEVICE"
 1637  echo "Unbinding..."
 1638  echo "$DEVICE" | sudo tee /sys/bus/usb/drivers/usb/unbind > /dev/null
 1639  sleep 2
 1640  echo "Rebinding..."
 1641  echo "$DEVICE" | sudo tee /sys/bus/usb/drivers/usb/bind > /dev/null
 1642  sleep 3
 1643  echo "Done. Camera should be ready."
 1644  lsusb | grep 3474
 1645  EOF
 1646  chmod +x ~/IgnisEdge/scripts/reset_p3.sh
 1647  ~/IgnisEdge/scripts/reset_p3.sh
 1648  sleep 5 && python ~/IgnisEdge/scripts/test_p3_classifier.py
 1649  ~/IgnisEdge/scripts/reset_p3.sh
 1650  sleep 5 && python ~/IgnisEdge/scripts/test_p3_classifier.py
 1651  python << 'EOF'
 1652  from pathlib import Path
 1653  script = Path.home() / "IgnisEdge/scripts/morphological_classifier.py"
 1654  text = script.read_text()
 1655  # Insert new rule BEFORE the AMBIGUOUS default fallback
 1656  old = """    # Rule 4: large halo + multiple sub-cores = wildfire (strong evidence)"""
 1657  new = """    # Rule 3.5: elongated flame with no halo and uniform temp = small flame source
 1658      # (encendedor, vela, mechero, soplete pequeño)
 1659      if (f.halo_extent < 0.5 and 
 1660          f.temp_gradient_c < 30.0 and 
 1661          2.0 < f.aspect_ratio < 4.0 and
 1662          f.area_px < 1000):
 1663          reasons.append(f"halo_ext={f.halo_extent:.2f}<0.5")
 1664          reasons.append(f"grad={f.temp_gradient_c:.1f}<30")
 1665          reasons.append(f"aspect={f.aspect_ratio:.1f} in [2,4]")
 1666          return ThreatClass.POINT_SOURCE, AlertLevel.NONE, "Small flame source: " + ", ".join(reasons)
 1667      # Rule 4: large halo + multiple sub-cores = wildfire (strong evidence)"""
 1668  if old in text:
 1669      text = text.replace(old, new)
 1670      script.write_text(text)
 1671      print("Patch aplicado: nueva regla 'small flame source' agregada")
 1672  else:
 1673      print("ERROR: no encontré la línea de Rule 4 para insertar antes")
 1674  EOF
 1675  grep -A 8 "Rule 3.5" ~/IgnisEdge/scripts/morphological_classifier.py
 1676  sleep 5 && python ~/IgnisEdge/scripts/test_p3_classifier.py
 1677  sudo snap install discord
 1678  sleep 5 && python ~/IgnisEdge/scripts/test_p3_classifier.py
 1679  cd /home/pablo-silva/IgnisEdge
 1680  cd ~/IgnisEdge && source venv/bin/activate
 1681  sleep 5 && python ~/IgnisEdge/scripts/test_p3_classifier.py 
 1682  source venv/bin/activate
 1683  python ~/IgnisEdge/scripts/capture_balcony_dataset.py
 1684  python scripts/autolabel_dataset.py
 1685  python ~/IgnisEdge/scripts/train_v3.py
 1686  python ~/IgnisEdge/scripts/capture_balcony_dataset.py
 1687  python edge_hybrid_node.py
 1688  python ~/IgnisEdge/scripts/edge_hybrid_node.py
 1689  cd ~/IgnisEdge/scripts
 1690  python live_inference.py
 1691  python hybrid_live_debug.py
 1692  python live_inference.py
 1693  python hybrid_live_debug.py
 1694  cd ~/IgnisEdge/scripts
 1695  python ignisedge_inspection_panel.py
 1696  history
