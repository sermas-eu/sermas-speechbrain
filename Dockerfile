FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /app
RUN mkdir /data

ENV CACHE_DIR=/cache
ENV TORCH_HOME=/cache/torch
ENV HF_HOME=/cache/hf
ENV SPEECHBRAIN_CACHE_DIR=/cache/speechbrain

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY sermas_speechbrain ./sermas_speechbrain

EXPOSE 5011

ENTRYPOINT [ "python3" ]
CMD ["-m", "flask", "--app", "sermas_speechbrain.api:app", "run", "--host=0.0.0.0", "--port=5011"]
