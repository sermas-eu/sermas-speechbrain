FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt update && apt install -y python3-pip

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD main.py .

RUN mkdir /data

EXPOSE 5011

CMD ["python", "main.py"]
