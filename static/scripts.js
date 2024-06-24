// scripts.js
document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');

    form.addEventListener('submit', function (event) {
        event.preventDefault();

        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => data[key] = value);

        fetch('/predict', {
            method: 'POST',
            body: new URLSearchParams(data)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultDiv.textContent = `Erro: ${data.error}`;
            } else {
                resultDiv.textContent = `Previsão: ${data.result} (Tempo de execução: ${data.execution_time.toFixed(2)} segundos)`;
            }
        })
        .catch(error => {
            resultDiv.textContent = `Erro: ${error}`;
        });
    });
});
