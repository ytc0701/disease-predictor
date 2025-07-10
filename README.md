# 疾病預測 API

使用 FastAPI 建置的疾病預測服務，提供網頁介面與 REST API。

## 執行

```bash
pip install -r requirements.txt
uvicorn main:app --reload

API 路徑
GET /：症狀表單介面

POST /predict：疾病預測 API