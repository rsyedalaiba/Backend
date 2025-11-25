from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client
import os

app = Flask(__name__)
CORS(app)

# Global Gradio client
client = None

def initialize_client():
    """Initialize Gradio client"""
    global client
    try:
        client = Client("syedalaibarehman/integrate")
        print("✅ Gradio client connected successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to connect Gradio client: {e}")
        return False

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "Python API is running!", 
        "gradio_connected": client is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("📨 Received data:", data)
        
        # Ensure client is connected
        if client is None:
            if not initialize_client():
                return jsonify({"success": False, "error": "Gradio client not connected"}), 500
        
        # Call prediction using Gradio client
        print("🔄 Calling Gradio prediction...")
        
        result = client.predict(
            data['province'],
            data['district'],
            data['crop_type'],
            data['soil_type'],
            data['sowing_date'],
            data['harvest_date'],
            float(data['area']),
            int(data['year']),
            float(data['temperature']),
            float(data['rainfall']),
            float(data['nitrogen']),
            float(data['phosphorus']),
            float(data['potassium']),
            float(data['soil_ph']),
            float(data['ndvi']),
            api_name="/predict_fn"
        )
        
        print("✅ Prediction completed:", result)
        
        return jsonify({
            "success": True, 
            "prediction": result,
            "source": "gradio-client"
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            "success": False, 
            "error": str(e)
        }), 500

@app.route('/api/direct_predict', methods=['POST'])
def direct_predict():
    """Alternative endpoint using direct API call"""
    try:
        data = request.json
        print("📨 Direct predict received:", data)
        
        # Create new client instance for this request
        temp_client = Client("syedalaibarehman/integrate")
        
        result = temp_client.predict(
            data['province'],
            data['district'], 
            data['crop_type'],
            data['soil_type'],
            data['sowing_date'],
            data['harvest_date'],
            float(data['area']),
            int(data['year']),
            float(data['temperature']),
            float(data['rainfall']),
            float(data['nitrogen']),
            float(data['phosphorus']),
            float(data['potassium']),
            float(data['soil_ph']),
            float(data['ndvi']),
            api_name="/predict_fn"
        )
        
        print("✅ Direct prediction completed:", result)
        
        return jsonify({
            "success": True,
            "prediction": result,
            "source": "direct-gradio-client"
        })
        
    except Exception as e:
        print(f"❌ Direct prediction error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Initialize client when server starts
@app.before_first_request
def setup():
    print("🚀 Initializing Gradio client...")
    initialize_client()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
