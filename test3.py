import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

sentence = input("digite a frase: ")
print(sentence)

test_sentences = [sentence]
print(test_sentences)

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(test_sentences)

sequences = tokenizer.texts_to_sequences(test_sentences)

max_length = max([len(seq) for seq in test_sentences])

test_padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')

model = model = keras.models.load_model('model.keras')
predictions = model.predict(test_padded_sequences)

# Imprimindo as predições
for sentence, prediction in zip(test_sentences, predictions):
    print(f'{sentence} - Sentimento: {"Positivo" if prediction >= 0.5 else "Negativo"} - Nivel de Confinaca: {prediction}')
