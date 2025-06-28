import gradio as gr
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

def predict_population(income, age, rooms, bedrooms, density, households, latitude):
    features = np.array([[income, age, rooms, bedrooms, density, households, latitude]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return f"""
        <div style='background: #f1f3f4; padding: 20px; border-radius: 12px; border-left: 5px solid #2e8b57; font-size: 22px;'>
            <strong>📈 Estimated Population:</strong> <span style='color:#2e8b57;'>{int(prediction):,}</span> people
        </div>
    """

with gr.Blocks(css="""
#main-container {
    max-width: 1100px;
    margin: auto;
    padding: 20px;
}
#title-card {
    padding: 30px 20px;
    background: linear-gradient(120deg, #d4fc79, #96e6a1);
    border-radius: 12px;
    text-align: center;
    margin-bottom: 30px;
}
#title-card h1 {
    font-size: 2.8rem;
    margin-bottom: 0.3rem;
}
#title-card p {
    font-size: 1.2rem;
    color: #333;
}
#predict-button {
    background-color: #2e8b57;
    color: white;
    font-weight: bold;
}
""") as demo:

    with gr.Column(elem_id="main-container"):
        gr.HTML("""
            <div id='title-card'>
                <h1>🏘️ California Population Predictor</h1>
                <p>Estimate population based on housing features using a trained Linear Regression model.</p>
            </div>
        """)

        with gr.Row():
            with gr.Column():
                income = gr.Slider(0.0, 15.0, value=3.0, label="💰 Median Income (in 10k units)")
                age = gr.Slider(1.0, 52.0, value=20.0, label="🏡 Housing Median Age")
                rooms = gr.Slider(100, 10000, value=3000, label="🚪 Total Rooms")
                bedrooms = gr.Slider(50, 2000, value=500, label="🛏️ Total Bedrooms")
            with gr.Column():
                density = gr.Slider(100, 20000, value=4000, label="👥 Population Density")
                households = gr.Slider(50, 2000, value=300, label="🏠 Households")
                latitude = gr.Slider(32.0, 42.0, value=36.0, label="🧭 Latitude")

        predict_btn = gr.Button("🔮 Predict Population", elem_id="predict-button")
        result_html = gr.HTML()

        predict_btn.click(
            fn=predict_population,
            inputs=[income, age, rooms, bedrooms, density, households, latitude],
            outputs=result_html
        )

demo.launch()