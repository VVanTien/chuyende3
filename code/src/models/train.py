from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
from src.config.config import MODEL_PATH

def train_accommodation_model(df):
    # Chọn các đặc trưng (features)
    # Lưu ý: Cần Encode các cột dạng chữ sang số
    le = LabelEncoder()
    df['Destination_enc'] = le.fit_transform(df['Destination'])
    df['Traveler_gender_enc'] = le.fit_transform(df['Traveler gender'])
    
    X = df[['Traveler age', 'Destination_enc', 'Traveler_gender_enc', 'Duration (days)']]
    y = df['Accommodation cost']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Lưu mô hình vào outputs/models/
    joblib.dump(model, MODEL_PATH)
    print(f"Mô hình đã được lưu tại {MODEL_PATH}")
    return model