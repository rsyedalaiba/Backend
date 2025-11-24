import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any

print("🚀 Starting Crop Yield Predictor...")

# -------------------------
# Import the inference functions
# -------------------------
try:
    from inference import predict_yield, PROVINCE_MAP, DISTRICT_MAP, CROP_MAP, SOIL_MAP
    print("✅ Inference module loaded successfully!")
except ImportError as e:
    print(f"❌ Could not load inference module: {e}")

# -------------------------
# Load model
# -------------------------
def load_model():
    try:
        model = joblib.load('model.joblib')
        print("✅ ACTUAL MODEL loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        # Fallback
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=100, n_features=15, random_state=42)
        fallback_model = RandomForestRegressor(n_estimators=10, random_state=42)
        fallback_model.fit(X, y)
        print("⚠️ Using fallback model")
        return fallback_model

model = load_model()

# -------------------------
# Gradio interface
# -------------------------
with gr.Blocks(title="Crop Yield Predictor") as demo:
    gr.Markdown("# 🌱 Crop Yield Prediction")
    gr.Markdown("Enter crop details for AI-powered yield prediction")
    
    with gr.Row():
        with gr.Column():
            province = gr.Dropdown(
                choices=list(PROVINCE_MAP.keys()),
                label="Province *",
                value="Punjab"
            )
            district = gr.Dropdown(
                choices=list(DISTRICT_MAP.keys()),
                label="District *",
                value="Lahore"
            )
            crop_type = gr.Dropdown(
                choices=list(CROP_MAP.keys()),
                label="Crop Type *",
                value="Wheat"
            )
            soil_type = gr.Dropdown(
                choices=list(SOIL_MAP.keys()),
                label="Soil Type *",
                value="Loamy"
            )
        
        with gr.Column():
            sowing_date = gr.Textbox(
                label="Sowing Date (MM/DD/YYYY) *",
                value="01/15/2024",
                placeholder="e.g., 01/15/2024 or 2024-01-15"
            )
            harvest_date = gr.Textbox(
                label="Harvest Date (MM/DD/YYYY) *",
                value="06/30/2024",
                placeholder="e.g., 06/30/2024 or 2024-06-30"
            )
            area = gr.Number(
                label="Area (Hectares) *",
                value=2.5,
                minimum=0.1
            )
            year = gr.Number(
                label="Year",
                value=2024,
                minimum=2000,
                maximum=2030
            )
    
    # Additional parameters
    with gr.Row():
        gr.Markdown("### Additional Parameters (Optional)")
    
    with gr.Row():
        temperature = gr.Number(label="Temperature (°C)", value=25.5)
        rainfall = gr.Number(label="Avg Rainfall (mm)", value=1200)
        nitrogen = gr.Number(label="Nitrogen (kg/ha)", value=150)
        phosphorus = gr.Number(label="Phosphorus (kg/ha)", value=80)
        potassium = gr.Number(label="Potassium (kg/ha)", value=100)
    
    with gr.Row():
        soil_ph = gr.Number(label="Soil pH", value=6.5)
        ndvi = gr.Number(label="NDVI", value=0.75)
    
    output = gr.Markdown(label="Prediction Result")
    predict_btn = gr.Button("Predict Yield", variant="primary")
    
    def predict_fn(
        province, district, crop_type, soil_type, sowing_date, harvest_date,
        area, year, temperature, rainfall, nitrogen, phosphorus, potassium,
        soil_ph, ndvi
    ):
        input_data = {
            'province': province,
            'district': district,
            'crop_type': crop_type,
            'soil_type': soil_type,
            'sowing_date': sowing_date,
            'harvest_date': harvest_date,
            'area': area,
            'year': year,
            'temperature': temperature,
            'rainfall': rainfall,
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'soil_ph': soil_ph,
            'ndvi': ndvi
        }
        try:
            result = predict_yield(input_data)
            if result['status'] == 'success':
                output_text = f"## 🏆 Predicted Yield: {result['formatted_yield']} {result['units']}\n\n"
                output_text += "### 📋 Recommendations:\n"
                for rec in result['recommendations']:
                    output_text += f"• {rec}\n"
                output_text += f"\n---\n*Powered by AI Model*"
                return output_text
            else:
                return f"## ❌ Prediction Error\n{result.get('error', 'Unknown error occurred')}"
        except Exception as e:
            return f"## ❌ System Error\n{str(e)}"
    
    predict_btn.click(
        predict_fn,
        inputs=[
            province, district, crop_type, soil_type, sowing_date, harvest_date,
            area, year, temperature, rainfall, nitrogen, phosphorus, potassium,
            soil_ph, ndvi
        ],
        outputs=output
    )

    # Examples section
    with gr.Accordion("📚 Example Inputs", open=False):
        gr.Markdown("""
**Example 1 (Wheat in Punjab):**
- Province: Punjab
- District: Lahore
- Crop Type: Wheat
- Soil Type: Loamy
- Sowing Date: 11/15/2023
- Harvest Date: 04/30/2024
- Area: 2.5 hectares

**Example 2 (Rice in Sindh):**
- Province: Sindh
- District: Karachi
- Crop Type: Rice
- Soil Type: Clay
- Sowing Date: 06/01/2024
- Harvest Date: 11/30/2024
- Area: 3.0 hectares
        """)

# Enable API for Lovable AI frontend
demo.config = demo.config.copy()
demo.config["show_api"] = True

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
