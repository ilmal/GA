FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY mnist_ps.py .

CMD [ "python3", "mnist_ps.py"]

