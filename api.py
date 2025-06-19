from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Caminho para o modelo e JSON com nomes das classes
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sports_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

# Carrega o modelo treinado
model = load_model(MODEL_PATH)

# Carrega os nomes das classes
with open(CLASSES_PATH, "r") as f:
    class_names = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')
from io import BytesIO
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', result="Nenhuma imagem enviada")

    img_file = request.files['image']
    if img_file.filename == '':
        return render_template('index.html', result="Nenhuma imagem selecionada")

    # Processa a imagem
    img = image.load_img(BytesIO(img_file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Faz a predição
    # Realiza a predição
    prediction = model.predict(img_array)[0]  # Remove dimensão extra

    # Pega os 3 índices com maior probabilidade
    top_indices = prediction.argsort()[-3:][::-1]

    # Monta a resposta com as 3 classes mais prováveis
    top_results = [
        f"{class_names[i]} ({prediction[i] * 100:.2f}%)"
        for i in top_indices
    ]

    # Junta os resultados para exibir na página
    result = " | ".join(top_results)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
