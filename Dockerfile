FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

WORKDIR /app

RUN python3 -m venv .venv

ADD requirements.txt .
RUN ./.venv/bin/pip3 install -r requirements.txt

ADD sermas_speechbrain ./sermas_speechbrain

RUN mkdir /data

EXPOSE 5011


ENV CACHE_DIR=/cache
ENV TORCH_HOME=/cache/torch
ENV HF_HOME=/cache/hf
ENV SPEECHBRAIN_CACHE_DIR=/cache/speechbrain

ENTRYPOINT [ "./.venv/bin/python3" ]
CMD ["-m", "flask", "--app", "sermas_speechbrain.api:app", "run", "--host=0.0.0.0", "--port=5011"]
