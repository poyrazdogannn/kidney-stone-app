import os
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pydicom
from PIL import Image

# Flask app
app = Flask(__name__)

# Modeli yükle
MODEL_PATH = "mobilenet_model.h5"   # kendi model dosyanı buraya koy
model = load_model(MODEL_PATH)

IMG_SIZE = (224, 224)  # eğitimde kullandığın boyut

# ---------------------------
# Yardımcı: DICOM okuma
# ---------------------------
def load_dicom_as_array(file):
    ds = pydicom.dcmread(file)
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))  # normalize 0-1
    arr = np.uint8(arr * 255)
    img = Image.fromarray(arr).convert("RGB")
    img = img.resize(IMG_SIZE)
    return img

# ---------------------------
# Grad-CAM fonksiyonu
# ---------------------------
def get_gradcam(img_array, model, class_index=0, layer_name=None):
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_out), axis=-1)
    heatmap = np.maximum(heatmap,0)/np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)
    return heatmap

@app.route("/", methods=["GET"])
def index():
    return "✅ Kidney Stone Model API çalışıyor! DICOM ve JPG/PNG destekleniyor."

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Dosya yüklenmedi"})
    file = request.files["file"]

    # DICOM veya normal resim yüklenmiş mi?
    if file.filename.lower().endswith(".dcm"):
        img = load_dicom_as_array(file)
    else:
        img = image.load_img(file, target_size=IMG_SIZE)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x)[0]
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))

    # Grad-CAM üret
    heatmap = get_gradcam(x, model, class_index=class_idx)
    grad_path = "gradcam.jpg"
    plt.imshow(img)
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.savefig(grad_path, bbox_inches="tight")
    plt.close()

    return jsonify({
        "prediction": int(class_idx),
        "confidence": confidence,
        "gradcam_url": "/gradcam"
    })

@app.route("/gradcam", methods=["GET"])
def gradcam_image():
    return send_file("gradcam.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)