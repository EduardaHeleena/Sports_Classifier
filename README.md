# Alunos: 
Eduarda Helena Silva 
João Vinicius 
# Classificador de Esportes com CNN

Este projeto consiste em um sistema de **classificação de imagens esportivas** utilizando uma rede neural convolucional (CNN), com interface web baseada em Flask.

## Estrutura do Projeto

```
sports_classifier/
├── data/                   # Conjunto de dados de treino, validação e teste
├── models/                 # Modelos treinados e arquivos de classes
│   ├── sports_model.h5
│   └── class_names.json
├── src/
│   └── api.py              # Código da API Flask
│    └── model.py 
│    └── predict.py
│    └── train.py 
├── templates/
│   └── index.html          # Interface web para envio de imagem
└── README.md
```

---

# Classificação de Imagens com CNN + EfficientNetB0

Este projeto consiste no desenvolvimento e treinamento de um modelo de **classificação de imagens esportivas** usando redes neurais convolucionais com **TensorFlow/Keras** e o modelo pré-treinado **EfficientNetB0**.

>  O modelo foi treinado no Google Colab.  
> 🔗 Acesse o notebook completo de treinamento neste link:  
> [Notebook de Treinamento no Google Colab](https://colab.research.google.com/drive/16l9TMhSuFyQUwHZ2kixeh_dhlo3fF_E-?usp=sharing)

---

## Dataset Utilizado

- **Fonte**: Kaggle — [Sports Classification Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification)
- **Formato**: Imagens organizadas em subpastas por categoria.
- **Processo de Download**:
  O dataset foi baixado diretamente no Google Colab com os comandos:

  ```bash
  !kaggle datasets download -d gpiosenka/sports-classification
  !unzip -q sports-classification.zip -d sports_images
  ```

---

## Pré-processamento e Aumento de Dados

- As imagens foram redimensionadas para **224x224 pixels**.
- Foi aplicado **data augmentation** apenas no conjunto de treino para melhorar a generalização do modelo:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
```

---

## Arquitetura do Modelo

Duas versões do modelo foram testadas:

### CNN Customizada

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

### Transfer Learning com EfficientNetB0

```python
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False  # Camadas congeladas inicialmente

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

---

## Compilação e Treinamento

O modelo foi compilado com os seguintes parâmetros:

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Early Stopping

Foi utilizada a técnica de **EarlyStopping** para evitar overfitting:

```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

### Treinamento

```python
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    callbacks=[early_stop]
)
```

---

## Avaliação

Após o treinamento, o modelo foi avaliado com os dados de teste por meio do `test_generator`, que não foi embaralhado para permitir análise posterior por índice.

---

## Salvamento do Modelo

O modelo final pode ser salvo para reutilização:

```python
model.save("modelo_esportes.h5")
```

---

### 4. Execute a API

```bash
python src/api.py
```

A aplicação rodará por padrão em [http://localhost:5000]

---

## Interface Web (index.html)

A página HTML permite que o usuário selecione e envie uma imagem para predição:

```html
<form action="/predict" method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <button type="submit">Enviar Imagem</button>
</form>
```

Se a imagem for válida, o modelo retorna as **três classes mais prováveis** com suas respectivas probabilidades.

---

## Funcionamento da API

Arquivo `src/api.py`:

1. **Rota `/`**: renderiza a interface web.
2. **Rota `/predict`**: recebe a imagem enviada e processa:
   - Redimensiona a imagem para 224x224.
   - Normaliza e prepara para entrada do modelo.
   - Executa a predição e retorna as 3 classes com maior probabilidade.

### Exemplo de resposta:

```
Predição: basketball (92.30%) | volleyball (4.22%) | cheerleading (2.01%)
```

---

## Observações

- Melhores resultados são obtidos com dataset balanceado.
- Imagens com baixa resolução ou contextos ambíguos podem gerar classificações imprecisas.

---
## Autores

Este projeto foi desenvolvido como parte de um estudo sobre classificação de imagens com redes neurais convolucionais (CNN) aplicadas ao reconhecimento de esportes.