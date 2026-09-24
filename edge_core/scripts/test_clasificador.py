import onnxruntime as ort
import numpy as np
import pandas as pd

session = ort.InferenceSession("ignisedge_v1.onnx")
CLASSES = ["fire", "zinc", "engine", "sun_metal", "person", "veg", "other"]

def predecir_imagen(ruta):
    crop = np.load(ruta).astype(np.float32)
    x = np.transpose(crop, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    logits = session.run(None, {"crop": x})[0][0]
    return CLASSES[np.argmax(logits)]

print("\n🔥 --- PRUEBA CON FUEGO REAL (Vela) --- 🔥")
df_vela = pd.read_csv("../dataset_vela15/manifest.csv")
fuegos = df_vela[df_vela["label"] == "fire"].sample(3)
for _, row in fuegos.iterrows():
    ruta = f"../dataset_vela15/{row['crop_path']}"
    print(f"Archivo: {row['crop_path']} | Real: Fuego -> Predicción IA: {predecir_imagen(ruta).upper()}")

print("\n🍳 --- PRUEBA CON CONFUSORES (Cocina) --- 🍳")
df_neg = pd.read_csv("../dataset_neg/manifest.csv")
cocina = df_neg.dropna(subset=["crop_path"]).sample(3)
for _, row in cocina.iterrows():
    ruta = f"../dataset_neg/{row['crop_path']}"
    print(f"Archivo: {row['crop_path']} | Real: Confusor -> Predicción IA: {predecir_imagen(ruta).upper()}")