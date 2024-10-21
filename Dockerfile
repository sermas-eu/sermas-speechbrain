FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

WORKDIR /app

# RUN apt update && apt install -y python3-pip

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD sermas_speechbrain ./sermas_speechbrain
ADD .env .

RUN mkdir /data

EXPOSE 5011

CMD ["python", "-m", "flask", "--app", "sermas_speechbrain.api:app", "run", "--host=0.0.0.0", "--port=5011"]
