import json
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

frases = [item['text'] for item in data]
labels = [item['label'] for item in data]

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(frases)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(frases)

max_length = max([len(seq) for seq in sequences])
print(max_length)
padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')

padded_sequences_np = np.array(padded_sequences)
labels_np = np.array(labels)

model = keras.Sequential([
    layers.Embedding(input_dim=len(word_index)+1, output_dim=128, input_length=max_length),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='tanh'),
    layers.Dense(64, activation='elu'),
    layers.Dense(32, activation='swish'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences_np, labels_np, epochs=1000)
model.save('model.keras')

# Testando o modelo
#test_sentences = ['Eu adoro essa música', 'Eu detesto esse cheiro', 'Eu amo essa comida', 'Eu amo essa musica', 'Eu amo esse carro']
test_sentences = [
    "Sinto-me em paz com minhas decisões.",
    "Cada passo me aproxima dos meus sonhos.",
    "Amo a sensação de superação após um desafio.",
    "O amor é a força mais poderosa do universo.",
    "Sinto-me preso em um ciclo sem fim.",
    "A tristeza parece não ter fim.",
    "Estou desapontado com o rumo das coisas.",                                                                             
    "A solidão é uma companhia constante.",
    "Sinto que a cada passo fico melhor em python",
    "Minha vida parece não fazer muito sentido ultimamente",
    "gostaria que as coisas fossem diferentes",
    "Que dia mais lindo",
    "estou muito feliz com o treinamento"
    ]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded_sequences = keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_length, padding='post')
predictions = model.predict(test_padded_sequences)
print(sequences)

# Imprimindo as predições
for sentence, prediction in zip(test_sentences, predictions):
    print(f'{sentence} - Sentimento: {"Positivo" if prediction >= 0.5 else "Negativo"} - Nivel de Confinaca: {prediction}')
