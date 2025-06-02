FROM apache/airflow:3.0.1

COPY requirements.txt /requirements.txt

# Although the image already has Airflow, we add it here to prevent pip from
# downgrading or upgrading apache airflow if there are dependency conflicts
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
