import gradio as gr
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "mobile_price_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "mobile_price_model.joblib was not found. "
        "Keep it in the same folder as app.py."
    )

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
model_name = bundle["model_name"]
features = bundle["features"]
price_labels = bundle["price_labels"]


def predict_price(ram, battery_power, fc, pc):
    values = [ram, battery_power, fc, pc]

    if any(v is None for v in values):
        raise gr.Error("Please enter values for all four features.")

    # Check that inputs are within the dataset's expected ranges.
    if not 256 <= ram <= 3998:
        raise gr.Error("RAM must be between 256 and 3998 MB.")
    if not 501 <= battery_power <= 1998:
        raise gr.Error("Battery power must be between 501 and 1998 mAh.")
    if not 0 <= fc <= 19:
        raise gr.Error("Front camera must be between 0 and 19 MP.")
    if not 0 <= pc <= 20:
        raise gr.Error("Primary camera must be between 0 and 20 MP.")

    data = pd.DataFrame(
        [[ram, battery_power, fc, pc]],
        columns=features
    )

    prediction = int(model.predict(data)[0])
    label = price_labels[prediction]

    result = f"""
## Predicted price category

# {label}

**Predicted class:** {prediction}

**Model:** {model_name}

*This is a category prediction, not an exact phone price.*
"""
    return result


css = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
}
#main-header {
    text-align: center;
    padding: 25px 10px;
    border-radius: 18px;
    background: linear-gradient(120deg, #172554, #2563eb);
    color: white;
    margin-bottom: 20px;
}
#main-header h1, #main-header p {
    color: white !important;
}
#prediction-output {
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    padding: 18px;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=css,
    title="Mobile Price Range Predictor"
) as demo:

    gr.HTML("""
    <div id="main-header">
        <h1>📱 Mobile Price Range Predictor</h1>
        <p>Machine Learning Project · Decision Tree vs Random Forest</p>
        <p>Enter mobile specifications to predict a price category.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ⚙️ Mobile Specifications")

            ram = gr.Slider(
                minimum=256, maximum=3998, value=2048, step=1,
                label="RAM (MB)"
            )
            battery_power = gr.Slider(
                minimum=501, maximum=1998, value=1500, step=1,
                label="Battery Power (mAh)"
            )
            fc = gr.Slider(
                minimum=0, maximum=19, value=8, step=1,
                label="Front Camera (MP)"
            )
            pc = gr.Slider(
                minimum=0, maximum=20, value=12, step=1,
                label="Primary Camera (MP)"
            )

            predict_btn = gr.Button(
                "Predict Price Category",
                variant="primary",
                size="lg"
            )

        with gr.Column():
            gr.Markdown("### 📊 Prediction Result")
            output = gr.Markdown(
                "Your prediction will appear here.",
                elem_id="prediction-output"
            )

    predict_btn.click(
        fn=predict_price,
        inputs=[ram, battery_power, fc, pc],
        outputs=output
    )

    gr.Markdown("""
    ---
    **About this project**

    This application uses a trained machine-learning classifier to predict
    one of four price categories: Low, Medium, High, or Very High.

    **Important:** The model uses only four features, so its predictions
    are estimates and may not reflect real-world market prices.
    """)

if __name__ == "__main__":
 demo.launch(server_name="0.0.0.0", server_port=int(__import__("os").environ.get("PORT", 10000)), share=False)
