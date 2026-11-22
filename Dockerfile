FROM python:3.12-slim

# CPU wheels by default; for an NVIDIA GPU build with
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
# (the CUDA wheels carry their own CUDA libraries; the host needs the NVIDIA runtime)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
# torchaudio 2.11 is the last release (maintenance mode); its compiled part matches
# torch 2.11 and the index's CPU/CUDA flavour, so both come from the same index
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0

ENV CACHE_DIR=/cache \
    TORCH_HOME=/cache/torch \
    HF_HOME=/cache/hf \
    SPEECHBRAIN_CACHE_DIR=/cache/speechbrain \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY constraints.txt .
RUN pip install --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
      -c constraints.txt "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"

COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

COPY sermas_speechbrain ./sermas_speechbrain

EXPOSE 5011

# waitress (production server), started by the module itself
CMD ["python3", "-m", "sermas_speechbrain.api"]
