import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB 
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score

# Exemplo de dados de texto
data = {
    'texto': [
        'A compra foi fraudulenta e o cartão foi clonado.',
        'Estou em desacordo com a cobrança na minha fatura.',
        'Meu carro foi roubado ontem à noite.',
        'A transação foi considerada fraude pelo banco.',
        'Há uma cobrança indevida na minha conta.',
        'Roubaram minha bicicleta na frente de casa.',
        'Recebi uma mensagem suspeita sobre meu cartão.',
        'Não reconheço essa compra na minha fatura.',
        'Alguém entrou em minha casa e roubou vários itens.'
    ],
    'categoria': [
        'Fraude', 'Desacordo', 'Roubo', 'Fraude', 'Desacordo', 'Roubo', 'Fraude', 'Desacordo', 'Roubo'
    ]
}

# Criar um DataFrame
df = pd.DataFrame(data)

# Separar os dados em variáveis de entrada e saída
X = df['texto']
y = df['categoria']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar um pipeline que inclua o TfidfVectorizer e o classificador MultinomialNB
model = make_pipeline(TfidfVectorizer(), MultinomialNB(), CategoricalNB(),  verbose=True)

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Testar o modelo com novos exemplos
novos_textos = [
    'Houve uma tentativa de fraude na minha conta bancária.',
    'Não concordo com esta taxa na minha fatura.',
    'Meu celular foi roubado na rua ontem.'
]
previsoes = model.predict(novos_textos)

for texto, categoria in zip(novos_textos, previsoes):
    print(f'Texto: {texto} -> Categoria: {categoria}')
