from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import json

# Carrega o modelo treinado
model = load_model("sports_model.h5")

# Carrega o mapeamento de classes
with open("class_names.json", "r") as f:
    class_names = json.load(f)

idx_to_class = {i: name for i, name in enumerate(class_names)}

def prever_imagem(caminho_imagem, modelo, tamanho=(224, 224), top_n=3):
    # Carrega e processa a imagem
    img = image.load_img(caminho_imagem, target_size=tamanho)
    img_array = image.img_to_array(img) / 255.0  # Mesmo pré-processamento usado no treinamento
    img_array = np.expand_dims(img_array, axis=0)

    # Realiza a predição
    pred = modelo.predict(img_array)[0]  # Remove dimensão extra

    # Top N predições
    top_indices = pred.argsort()[-top_n:][::-1]
    top_preds = [(idx_to_class[i], pred[i] * 100) for i in top_indices]

    # Exibe a imagem com o título principal sendo a classe mais provável
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{top_preds[0][0]} ({top_preds[0][1]:.2f}%)", fontsize=14)
    plt.show()

    # Imprime todas as N melhores predições
    for nome, score in top_preds:
        print(f"{nome}: {score:.2f}%")

    return top_preds
