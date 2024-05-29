import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

with open('category_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

text = [item['text'] for item in data]
category = [item['category'] for item in data]

# Exemplo de dados de texto
data = {
    'text': text,
    'category': category
}

# Criar um DataFrame
data_frame = pd.DataFrame(data)

# Separar os dados em variáveis de entrada (X) e saída (y)
X = data_frame['text']
y = data_frame['category']

# Codificar as etiquetas de saída
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = keras.utils.to_categorical(y_encoded)
print(f'Categorias encontradas: {y_encoded}')
print(f'Vetores das categorias: {y_categorical}')

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Tokenizar os textos
tokenizer = keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
print(f'Tokens encontrados: {word_index}')

# Converter textos em sequências de inteiros
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Padronizar o comprimento das sequências
max_length = 50
X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train_sequences, maxlen=max_length, padding='post')
X_test_padded = keras.preprocessing.sequence.pad_sequences(X_test_sequences, maxlen=max_length, padding='post')

#Embedding Layer: Converte palavras em vetores densos.
#SpatialDropout1D Layer: Reduz overfitting.
#Conv1D Layer: Extrai características locais dos textos.
#MaxPooling1D Layer: Reduz a dimensionalidade das características extraídas.
#Bidirectional LSTM: Captura dependências temporais em ambas as direções.
#GlobalMaxPooling1D Layer: Reduz dimensionalidade, mantendo características mais relevantes.
#Dense Layer: Processa características extraídas.
#Dropout Layer: Reduz overfitting.
#BatchNormalization Layer: Normaliza ativações para melhorar a convergência.
#Output Dense Layer: Classifica os textos nas categorias desejadas.

# Construir o modelo
model = Sequential([
    layers.Embedding(input_dim=len(word_index)+1, output_dim=128, input_length=max_length),
    layers.SpatialDropout1D(0.5),
    layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Bidirectional(layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
    layers.GlobalMaxPooling1D(),
    layers.Dense(128, activation='softplus'),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(3, activation='softmax')
])

# Compilar o modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train_padded, y_train, epochs=200, batch_size=32, validation_data=(X_test_padded, y_test), verbose=2)

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=2)
print(f'Accuracy: {accuracy}')

# Testar o modelo com novos exemplos
novos_textos = [
    'Houve uma tentativa de fraude na minha conta bancária.',
    'Não concordo com esta taxa na minha fatura.',
    'Meu celular foi roubado na rua ontem.'
]

# Pré-processar novos textos
novos_textos_sequences = tokenizer.texts_to_sequences(novos_textos)
novos_textos_padded = keras.preprocessing.sequence.pad_sequences(novos_textos_sequences, maxlen=max_length, padding='post')

# Fazer previsões
previsoes = model.predict(novos_textos_padded)
previsoes_classes = np.argmax(previsoes, axis=1)

# Mapear previsões de volta para as categorias originais
previsoes_categorias = label_encoder.inverse_transform(previsoes_classes)

for texto, categoria in zip(novos_textos, previsoes_categorias):
    print(f'Texto: {texto} -> Categoria: {categoria}')
