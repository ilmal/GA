FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY mnist_worker.py .

# '{"cluster": {"worker": ["192.168.3.38:5412", "192.168.3.229:5412"]}, "task": {"type": "worker", "index": 0} }'

#ARG tf_config=default_value

#ENV TF_CONFIG=$tf_config

CMD [ "python3", "mnist_worker.py"]

