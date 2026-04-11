# Streamlit Frontend

## 1. Install dependencies

Ensure backend dependencies are installed, then install frontend additions:

```bash
pip install streamlit requests
```

Or use `requirements.txt` after it is updated.

## 2. Start backend

```bash
uvicorn app.main:app --reload
```

## 3. Start frontend

```bash
streamlit run frontend/app.py
```

## 4. Optional backend URL

Default backend URL is `http://127.0.0.1:8000`.
You can override it with:

```bash
set BACKEND_BASE_URL=http://127.0.0.1:8000
streamlit run frontend/app.py
```
