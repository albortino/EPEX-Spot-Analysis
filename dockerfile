FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

COPY . .

EXPOSE 8501

# Replace 'epex-analysis' with the name from your environment.yml
CMD ["conda", "run", "-n", "epex-analysis", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.maxUploadSize=5"]