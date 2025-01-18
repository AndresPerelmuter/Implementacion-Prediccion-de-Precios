import io
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# Cargar el modelo y el dataset al inicio
modelo = joblib.load('best_model_dt.pkl')

# Cargar el dataset y calcular la media de precios al inicio
df = pd.read_csv('df.csv.gz', compression='gzip', nrows=100000)
mean_precio_real = df['price'].mean()  # Calcular una sola vez la media

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Recoger datos del formulario
    datos_usuario = {
        'host_response_rate': float(request.form['host_response_rate']),
        'host_acceptance_rate': float(request.form['host_acceptance_rate']),
        'accommodates': int(request.form['accommodates']),
        'bathrooms_count': float(request.form['bathrooms_count']),
        'bedrooms': int(request.form['bedrooms']),
        'availability_30': int(request.form['availability_30']),
        'availability_60': int(request.form['availability_60']),
        'availability_90': int(request.form['availability_90']),
        'availability_365': int(request.form['availability_365']),
        'number_of_reviews': int(request.form['number_of_reviews']),
        'review_scores_rating': float(request.form['review_scores_rating']),
        'review_scores_value': float(request.form['review_scores_value']),
        'reviews_per_month': float(request.form['reviews_per_month']),
        'shared': int(request.form['shared']),
        'host_is_superhost_encoded': int(request.form['host_is_superhost_encoded']),
        'room_type_encoded': int(request.form['room_type_encoded']),
        'month': int(request.form['month']),
        'antiguedad': int(request.form['antiguedad']),
    }

    # Crear DataFrame con los datos del usuario
    data = pd.DataFrame([datos_usuario])

    # Hacer la predicción
    log_precio_estimado = modelo.predict(data)[0]
    precio_estimado = round(np.exp(log_precio_estimado), 2)  # Convertir de logaritmo a precio original

    # Filtrar el DataFrame por accommodates = 4 y month = 1
    df_filtrado = df[(df['accommodates'] == 4) & (df['month'] == 1)]
    df_filtrado['price_original'] = np.exp(df_filtrado['price'])  # Convertir log(price) a precio original

    # Crear el gráfico de densidad unidimensional
    fig = px.histogram(df_filtrado, x="price", title='Densidad de Precios')
    fig.add_vline(x=precio_estimado, line_width=3, line_dash="dash", line_color="red",
                  annotation_text="Precio estimado", annotation_position="top right")

    # Convertir la figura a HTML y enviarla al HTML
    graph_html = pio.to_html(fig, full_html=False)

    return render_template(
        'index.html',
        prediccion=f"{precio_estimado:.2f}",
        graph_html=graph_html,
        datos_usuario=datos_usuario  # Pasar los datos ingresados al template
    )


@app.route('/plot', methods=['POST'])
def plot():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(title="Ejemplo de Gráfico", xlabel="Eje X", ylabel="Eje Y")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)

    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
