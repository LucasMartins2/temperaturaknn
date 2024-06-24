import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Carregar o arquivo CSV com os dados
df = pd.read_csv('base_teste.csv')

# Tratar valores ausentes, se necessário
df = df.dropna()

# Converter a coluna 'Data' para o formato datetime
df['Data'] = pd.to_datetime(df['Data'])

# Categorizar a temperatura
def categorize_temperature(temp):
    if temp > 25:
        return 'quente'
    elif 18 < temp <= 25:
        return 'morna'
    else:
        return 'fria'

df['Categoria'] = df['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].apply(categorize_temperature)

# Selecionar as features e o target
X = df[['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)', 
        'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)', 
        'RADIACAO GLOBAL (Kj/m²)', 
        'TEMPERATURA DO PONTO DE ORVALHO (°C)', 
        'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)', 
        'UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)', 
        'UMIDADE RELATIVA DO AR, HORARIA (%)', 
        'VENTO, RAJADA MAXIMA (m/s)', 
        'VENTO, VELOCIDADE HORARIA (m/s)']]
y = df['Categoria']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar o modelo KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred = model.predict(X_test_scaled)

# Avaliar o modelo
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Salvar o modelo e o scaler
joblib.dump(model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
