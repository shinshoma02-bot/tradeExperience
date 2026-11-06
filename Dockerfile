FROM python:3.13-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY keiba/site ./keiba/site
COPY boat/site ./boat/site
COPY pachinko/site ./pachinko/site
ENV PORT=8080
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT}"]
