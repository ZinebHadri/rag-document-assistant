FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["sh", "-c", "python -m uvicorn api:app --host 0.0.0.0 --port 8000 & python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.fileWatcherType none"]