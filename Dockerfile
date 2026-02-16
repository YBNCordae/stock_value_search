FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CloudBase / 容器平台一般用 PORT 环境变量
EXPOSE 8501
CMD ["bash", "-lc", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
