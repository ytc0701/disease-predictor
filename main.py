'''建立API'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

from fastapi.templating import Jinja2Templates  # 新增
from fastapi.responses import HTMLResponse      # 新增
from fastapi import Request                     # 新增

    
# === 建立 FastAPI 應用 ===
app = FastAPI(title="疾病預測 API", description="輸入症狀特徵，回傳預測疾病", version="1.0")

# 新增：設定模板資料夾
templates = Jinja2Templates(directory="templates")

# 新增首頁路由
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# === 定義請求資料格式 ===
class SymptomInput(BaseModel):
    symptoms: dict

# === 疾病中英文對照表 ===
disease_name_mapping = {
    "(vertigo) Paroymsal  Positional Vertigo": "姿勢性暈眩",
    "AIDS": "愛滋病",
    "Acne": "痤瘡",
    "Alcoholic hepatitis": "酒精性肝炎",
    "Allergy": "過敏",
    "Arthritis": "關節炎",
    "Bronchial Asthma": "支氣管哮喘",
    "Cervical spondylosis": "頸椎病",
    "Chicken pox": "水痘",
    "Chronic cholestasis": "慢性膽汁鬱積",
    "Common Cold": "感冒",
    "Dengue": "登革熱",
    "Diabetes": "糖尿病",
    "Dimorphic hemmorhoids(piles)": "混合痔瘡",
    "Drug Reaction": "藥物反應",
    "Fungal infection": "真菌感染",
    "GERD": "胃食道逆流",
    "Gastroenteritis": "腸胃炎",
    "Heart attack": "心臟病發作",
    "Hepatitis B": "B型肝炎",
    "Hepatitis C": "C型肝炎",
    "Hepatitis D": "D型肝炎",
    "Hepatitis E": "E型肝炎",
    "Hypertension": "高血壓",
    "Hyperthyroidism": "甲狀腺功能亢進",
    "Hypoglycemia": "低血糖",
    "Hypothyroidism": "甲狀腺功能低下",
    "Impetigo": "膿痂疹",
    "Jaundice": "黃疸",
    "Malaria": "瘧疾",
    "Migraine": "偏頭痛",
    "Osteoarthristis": "骨關節炎",
    "Paralysis (brain hemorrhage)": "癱瘓（腦出血）",
    "Peptic ulcer diseae": "胃潰瘍",
    "Pneumonia": "肺炎",
    "Psoriasis": "乾癬",
    "Tuberculosis": "結核病",
    "Typhoid": "傷寒",
    "Urinary tract infection": "尿道感染",
    "Varicose veins": "靜脈曲張",
    "hepatitis A": "A型肝炎"
}
# === 前處理函式 ===
def preprocess_input(symptom_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([symptom_dict])

    # 清理欄位
    df = df.dropna(axis=1, how='all')
    if "fluid_overload" in df.columns:
        df = df.drop(columns=["fluid_overload"])
    if "fluid_overload.1" in df.columns:
        df = df.rename(columns={"fluid_overload.1": "fluid_overload"})

    # 定義 feature group（跟你原來的完全一樣）
    feature_groups = [
        (['nodal_skin_eruptions', 'dischromic_patches'], ['Fungal infection']),
        (['shivering', 'watering_from_eyes'], ['Allergy']),
        (['muscle_wasting', 'patches_in_throat', 'extra_marital_contacts'], ['AIDS']),
        (['weight_gain', 'cold_hands_and_feets', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties'], ['Hypothyroidism']),
        (['anxiety', 'drying_and_tingling_lips', 'slurred_speech', 'palpitations'], ['Hypoglycemia']),
        (['mood_swings', 'abnormal_menstruation'], ['Hyperthyroidism', 'Hypothyroidism']),
        (['irregular_sugar_level', 'increased_appetite', 'polyuria'], ['Diabetes ']),
        (['sunken_eyes', 'dehydration'], ['Gastroenteritis']),
        (['yellow_urine', 'receiving_blood_transfusion', 'receiving_unsterile_injections'], ['Hepatitis B']),
        (['acute_liver_failure', 'coma', 'stomach_bleeding'], ['Hepatitis E']),
        (['swelling_of_stomach', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload'], ['Alcoholic hepatitis']),
        (['throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'loss_of_smell'], ['Common Cold']),
        (['pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus'], ['Dimorphic hemmorhoids(piles)']),
        (['cramps', 'bruising', 'swollen_legs', 'swollen_blood_vessels', 'prominent_veins_on_calf'], ['Varicose veins']),
        (['knee_pain', 'hip_joint_pain'], ['Osteoarthristis']),
        (['swelling_joints', 'painful_walking'], ['Arthritis', 'Osteoarthristis']),
        (['spinning_movements', 'unsteadiness'], ['(vertigo) Paroymsal  Positional Vertigo']),
        (['weakness_of_one_body_side', 'altered_sensorium'], ['Paralysis (brain hemorrhage)']),
        (['bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine'], ['Urinary tract infection']),
        (['passage_of_gases', 'internal_itching'], ['Peptic ulcer diseae']),
        (['toxic_look_(typhos)', 'belly_pain'], ['Typhoid']),
        (['pus_filled_pimples', 'blackheads', 'scurring'], ['Acne']),
        (['skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails'], ['Psoriasis']),
        (['blister', 'red_sore_around_nose', 'yellow_crust_ooze'], ['Impetigo']),
    ]

    # 新增 _related 欄位
    all_used_features = set()
    for features, diseases in feature_groups:
        existing_features = [f for f in features if f in df.columns]
        if not existing_features:
            continue
        col_name = diseases[0].strip() + "_related"
        df[col_name] = df[existing_features].max(axis=1)
        all_used_features.update(existing_features)

    # 移除原始特徵欄位
    df = df.drop(columns=list(all_used_features), errors='ignore')

    return df

# === 載入模型與編碼器 ===
try:
    model = joblib.load("disease_prediction_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    features = list(model.feature_names_in_)  # 模型所需特徵名稱
except Exception as e:
    raise RuntimeError("❌ 模型或編碼器載入失敗：", e)
    
    
# === 預測 API ===
@app.post("/predict")
def predict_disease(data: SymptomInput):
    try:
        df_processed = preprocess_input(data.symptoms)

        for col in model.feature_names_in_:
            if col not in df_processed.columns:
                df_processed[col] = 0
        df_processed = df_processed[model.feature_names_in_]

        prediction_encoded = model.predict(df_processed)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        zh_name = disease_name_mapping.get(prediction_label.strip(), "未知疾病")

        return {
            "prediction_zh": zh_name,
                "prediction_en": prediction_label.strip(),
                "encoded_label": int(prediction_encoded)
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
