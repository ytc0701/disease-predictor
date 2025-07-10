import joblib

try:
    model = joblib.load("disease_prediction_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    print("✅ 模型與編碼器載入成功！")
    print("➡️ 模型輸入特徵數量：", len(model.feature_names_in_))

except Exception as e:
    print("❌ 發生錯誤：", e)