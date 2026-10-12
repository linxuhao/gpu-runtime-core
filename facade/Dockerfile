FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py engine_routes.py ./
USER 1000:1000
CMD ["python", "app.py"]
