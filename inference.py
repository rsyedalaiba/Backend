import pandas as pd
from datetime import datetime
import joblib
import os

# Your exact mappings from the provided data
PROVINCE_MAP = {
    "Balochistan": 0,
    "Balochitan": 1, 
    "KPK": 2,
    "Punjab": 3,
    "Sindh": 4
}

DISTRICT_MAP = {
    # Punjab (40+ districts)
    "Lahore": 99, "Faisalabad": 37, "Rawalpindi": 156, "Multan": 124,
    "Gujranwala": 42, "Sialkot": 176, "Bahawalpur": 10, "Sargodha": 166,
    "Jhang": 62, "D.G. Khan": 26, "Rahim Yar Khan": 152, "Sheikhupura": 170,
    "Gujrat": 43, "Kasur": 79, "Sahiwal": 162, "Okara": 140,
    "M.B. Din": 112, "Toba Tek Singh": 192, "Jhelum": 64, "Chakwal": 21,
    "Khushab": 86, "Mianwali": 120, "Bhakkar": 16, "Layyah": 105,
    "Vehari": 196, "Khanewal": 82, "Pakpattan": 143, "Bahawalnagar": 8,
    "Rajanpur": 154, "Lodhran": 107, "Nankana Sahib": 133, "Chiniot": 24,
    "Hafizabad": 48, "M. Garh": 110, "Attock": 2, "Talagang": 185,
    
    # Sindh (30+ districts)
    "Karachi": 75, "Hyderabad": 54, "Sukkur": 179, "Larkana": 101,
    "Mirpurkhas": 121, "Jacobabad": 57, "Khairpur": 80, "Thatta": 190,
    "Badin": 6, "Sanghar": 164, "N. Feroze": 130, "Ghotki": 40,
    "Shaheed Benazir Abad": 168, "Shikarpur": 174, "Dadu": 31, "Umarkot": 195,
    "Tharparkar": 188, "Kashmore": 78, "Jamshoro": 60, "Matiari": 119,
    "Tando Allahyar": 186, "Jaffarabad": 59, "Nasirabad": 136,
    
    # KPK (30+ districts)
    "Peshawar": 147, "Mardan": 116, "Abbottabad": 1, "Swat": 182,
    "Bannu": 13, "Kohat": 93, "Charsadda": 23, "Nowshera": 138,
    "Mansehra": 115, "Swabi": 181, "D.I. Khan": 28, "Haripur": 52,
    "Malakand": 114, "Karak": 77, "Lakki Marwat": 100, "Tank": 187,
    "Dir Lower": 35, "Dir Upper": 36, "Buner": 18, "Shangla": 169,
    "Kohistan": 94, "Khyber": 89, "Bajour": 12, "Mohmand": 122,
    "Kurram": 98, "Hangu": 50, "Orakzai": 141, "N. Waziristan": 132,
    "S. Waziristan": 157, "Chitral": 25, "Battagram": 15,
    
    # Balochistan (30+ districts)
    "Quetta": 149, "Khuzdar": 87, "Gwadar": 46, "Sibi": 177,
    "Loralai": 108, "Zhob": 199, "Killa Abdullah": 90, "Killa Saifullah": 91,
    "Pishin": 148, "Chagai": 20, "Mastung": 117, "Kalat": 71,
    "Kachhi": 68, "Nushki": 139, "Jhal Magsi": 61, "Awaran": 4,
    "Lasbela": 103, "Barkhan": 14, "Musa Khail": 128, "Kohlu": 95,
    "Dera Bugti": 34, "Ziarat": 201, "Kharan": 84, "Panjgur": 146,
    "Washuk": 197, "Harnai": 53, "Sherani": 173,
    
    # Islamabad & Others
    "Islamabad": 56
}

CROP_MAP = {
    "Cotton": 0,
    "Maize": 1, 
    "Potato": 2,
    "Rice": 3,
    "Sugarcane": 4,
    "Tomato": 5,
    "Wheat": 6
}

SOIL_MAP = {
    "Clay": 1,
    "Sandy": 2, 
    "Loamy": 3,
    "Silty": 4
}

# Load your model (adjust based on your model format)
def load_model():
    try:
        # If you have a .pkl file
        model = joblib.load('model.joblib')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Return a dummy model for testing
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor()

model = load_model()

def date_to_season_code(date_str):
    """Convert any date format to season code (1-4)"""
    try:
        # Handle different date formats
        if '/' in date_str:
            parts = date_str.split('/')
            if len(parts) >= 2:
                month = int(parts[0])
            else:
                return 1
        elif '-' in date_str:
            parts = date_str.split('-')
            if len(parts) >= 2:
                month = int(parts[1])  # Assuming YYYY-MM-DD format
            else:
                return 1
        else:
            # Try to parse as full date string
            try:
                date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                month = date_obj.month
            except:
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    month = date_obj.month
                except:
                    return 1
        
        # Convert month to season
        if month in [12, 1, 2]:
            return 1  # Winter
        elif month in [3, 4, 5]:
            return 2  # Spring  
        elif month in [6, 7, 8]:
            return 3  # Summer
        else:
            return 4  # Fall
            
    except Exception as e:
        print(f"Date parsing error for '{date_str}': {e}")
        return 1  # Default to Winter

def preprocess_input(data):
    return {
        "Province_encoded": PROVINCE_MAP.get(data.get('province', 'Punjab'), 3),
        "District_encoded": DISTRICT_MAP.get(data.get('district', 'Lahore'), 99),
        "Crop_Type_encoded": CROP_MAP.get(data.get('crop_type', 'Wheat'), 6),  # ADD THIS!
        "Soil_Type_encoded": SOIL_MAP.get(data.get('soil_type', 'Loamy'), 3),
        "Sowing_Time_encoded": date_to_season_code(data.get('sowing_date', '01/01/2024')),
        "Harvest_time_encoded": date_to_season_code(data.get('harvest_date', '12/31/2024')),
        "Year": int(data.get('year', 2024)),
        "Area_in_Hectares": float(data.get('area', 2.5)),  # UNDERSCORE
        "Temperature(C)": float(data.get('temperature', 25.5)),
        "Avg_Rainfall_(mm)": float(data.get('rainfall', 1200)),  # UNDERSCORE
        "N_(kg/ha)": float(data.get('nitrogen', 150)),  # UNDERSCORE
        "P_(kg/ha)": float(data.get('phosphorus', 80)),  # UNDERSCORE  
        "K_(kg/ha)": float(data.get('potassium', 100)),  # UNDERSCORE
        "Soil_pH": float(data.get('soil_ph', 6.5)),  # UNDERSCORE
        "NDVI": float(data.get('ndvi', 0.75))
    }

def predict_yield(data):
    """Main prediction function"""
    try:
        # Preprocess input
        processed_data = preprocess_input(data)
        
        # Convert to DataFrame for model prediction
        input_df = pd.DataFrame([processed_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Generate recommendations
        recommendations = []
        if prediction < 100000:
            recommendations = [
                "Consider soil testing for nutrient optimization",
                "Optimize irrigation schedule based on crop stage", 
                "Review fertilizer application timing"
            ]
        elif prediction > 150000:
            recommendations = [
                "Maintain current agricultural practices",
                "Monitor for potential pest outbreaks",
                "Ensure timely harvesting"
            ]
        else:
            recommendations = [
                "Good yield expected with current practices",
                "Continue regular monitoring",
                "Consider crop rotation for soil health"
            ]
        
        return {
            "predicted_yield": round(float(prediction), 2),
            "formatted_yield": f"{prediction:,.2f}",
            "units": "units",
            "recommendations": recommendations,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

# For Hugging Face Inference API
def inference(data):
    return predict_yield(data)