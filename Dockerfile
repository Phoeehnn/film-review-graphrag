FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY requirements-service.txt /app/
RUN pip install --no-cache-dir torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-service.txt
COPY service /app/service
COPY scripts /app/scripts
COPY pages /app/pages
COPY ui /app/ui
COPY .streamlit /app/.streamlit
COPY docs /app/docs
COPY app.py graph_chain.py config.py /app/
EXPOSE 8000 8501
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
