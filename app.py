import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

app = Flask(__name__)

# Carregar os dados
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

# Selecionar as features (variáveis independentes) e o target (variável dependente)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

@app.route('/')
def index():
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    novos_dados = [float(request.form['precipitacao']),
                   float(request.form['pressao']),
                   float(request.form['radiacao']),
                   float(request.form['temp_orvalho']),
                   float(request.form['umidade_max']),
                   float(request.form['umidade_min']),
                   float(request.form['umidade_rel']),
                   float(request.form['vento_rajada']),
                   float(request.form['vento_velocidade'])]
    
    novos_dados_scaled = scaler.transform([novos_dados])
    prediction = model.predict(novos_dados_scaled)[0]
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    return render_template('index.html', prediction=prediction, execution_time=execution_time, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
