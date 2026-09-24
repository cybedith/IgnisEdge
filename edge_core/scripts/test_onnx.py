import onnxruntime as ort
import numpy as np
from pathlib import Path

# 1. Cargar el cerebro ONNX
ort_session = ort.InferenceSession("ignisedge_v1.onnx")

# 2. Buscar un archivo .npy de fuego real en tu carpeta
crop_path = next(Path("../dataset_vela15/crops").glob("*.npy"))
crop = np.load(crop_path).astype(np.float32)

# 3. Darle el formato que espera la red neuronal
# El recorte original es (64, 64, 3). Lo pasamos a (1, 3, 64, 64)
x = np.transpose(crop, (2, 0, 1))
x = np.expand_dims(x, axis=0)

# 4. Ejecutar la inferencia
outputs = ort_session.run(None, {"crop": x})
clase_predicha = np.argmax(outputs[0], axis=1)[0]

# En train_classifier.py, "fire" es el índice 0.
if clase_predicha == 0:
    print(f"¡Éxito! El dron detectó FUEGO en {crop_path.name}")
else:
    print(f"Fallo. El modelo lo clasificó como clase {clase_predicha}.")