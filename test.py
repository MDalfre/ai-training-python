import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Definindo os dados de treinamento e teste
sentences = ['Eu amo essa comida', 'Eu odeio esse filme', 'Eu gosto de andar no parque', 'Eu detesto esperar no trânsito']
labels = [1, 0, 1, 0]

# Convertendo os textos para vetores numéricos
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)

# Padding dos vetores numéricos
max_length = max([len(seq) for seq in sequences])
padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')

padded_sequences_np = np.array(padded_sequences)
labels_np = np.array(labels)

# Criando o modelo
model = keras.Sequential([
    layers.Embedding(len(word_index)+1, 16, input_length=max_length),
    layers.GlobalAveragePooling1D(),
    layers.Dense(1, activation='sigmoid')
])

# Compilando o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Comeca Aqui")

# Treinando o modelo
model.fit(padded_sequences_np, labels_np, epochs=10000)

# Testando o modelo
test_sentences = ['Eu adoro essa música', 'Eu detesto esse cheiro', 'Eu amo essa comida', 'Eu amo essa musica', 'Eu amo esse carro']
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded_sequences = keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_length, padding='post')
predictions = model.predict(test_padded_sequences)

# Imprimindo as predições
for sentence, prediction in zip(test_sentences, predictions):
    print(f'{sentence} - Sentimento: {"Positivo" if prediction >= 0.5 else "Negativo"} - Nivel de Confinaca: {prediction}')

