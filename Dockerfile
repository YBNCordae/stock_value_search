FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
CMD ["bash", "-lc", "streamlit run app.py --server.address=0.0.0.0 --server.port=80"]
