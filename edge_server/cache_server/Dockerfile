#FROM tiangolo/uwsgi-nginx-flask:python3.10
#FROM rasoul5g/drl_cache_app:latest
FROM python
COPY ./requirements.txt /app/requirements.txt

RUN #pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip install  --upgrade -r /app/requirements.txt

ARG CACHEBUST=1
COPY ./app /app
RUN mkdir -p /videos
CMD ["bash", "-c","python3 /app/main.py"]