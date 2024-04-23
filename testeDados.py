import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri)

x= dados[["home","how_it_works","contact"]]
y= dados[["bought"]]

treino_x = x[:75]
treino_y = y[:75]
teste_x = x[75:]
teste_y = y[75:]

print(f"Treinaremos com {len(treino_x)} elementos e testaremos com {len(teste_x)} elementos")

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

model = LinearSVC()
model.fit(treino_x,treino_y)
previsoes = modelo.predict(teste_x)

taxa_acerto = accuraccy_score (teste_y, previsoes) * 100

print(f"Acurácia foi de {taxa_acerto}")