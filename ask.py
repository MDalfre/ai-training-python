import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Carregar o modelo primeiro para obter o comprimento esperado
model = keras.models.load_model('model.keras')

# Obtendo o comprimento esperado das sequências de entrada a partir da camada de entrada do modelo
expected_length = model.layers[0].input_shape[1]
print(f'esse modelo possui {expected_length} de tamanho maximo')

# Entrada do usuário
sentence = input("Digite a frase: ")
print(sentence)

test_sentences = [sentence]
print(test_sentences)

# Tokenização
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(test_sentences)
print(tokenizer.index_word)

sequences = tokenizer.texts_to_sequences(test_sentences)
print(sequences)
print(f'Esta frase tem {len(sequences)} de tamanho')

# Ajustar o comprimento das sequências para o esperado pelo modelo
test_padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=expected_length, padding='post')

# Fazer predições
predictions = model.predict(test_padded_sequences)

# Imprimir as predições
for sentence, prediction in zip(test_sentences, predictions):
    print(f'{sentence} - Sentimento: {"Positivo" if prediction >= 0.5 else "Negativo"} - Nível de Confiança: {prediction[0]}')
