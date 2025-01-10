import base64
from io import BytesIO
import os
import pathlib
import pickle

import dotenv
import numpy as np
import pyannote.audio
import torch
from speechbrain.inference import classifiers, interfaces, separation
import os
from sermas_speechbrain import _core

_root_dir = pathlib.Path(__file__).parents[1]
dotenv.load_dotenv(_root_dir / ".env")  # This loads .env values as environment variable


# TODO: This can be a lot cleaner
run_opts = {
    # enable CUDA
    "device": "cuda",
    "data_parallel_count": -1,
    "data_parallel_backend": False,
    "distributed_launch": False,
    "distributed_backend": "nccl",
    "jit_module_keys": None,
}

# defaults to GPU
if "USE_GPU" in os.environ.keys() and os.environ["USE_GPU"] == "0":
    print("Using CPU")
    run_opts = {}
else:
    print("Using GPU")

################
# Denoising
################
_denoiser = separation.SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-whamr-enhancement",
    savedir="/app/speechbrain_models/sepformer-whamr-enhancement",
)

###############
# Diarization
###############
_diarization_service = os.environ.get("DIARIZATION_SERVICE", "").lower()
if _diarization_service == "local":

    def _get_speaker_count(audio: _core.Audio) -> dict:
        # TODO: This is NOT a speaker counter but rather a noise checker.
        # basically, if the background signal is too loud, it will classify two speakers
        # probability values are just placeholders
        np_waveform = audio.waveform.numpy()
        silence_threshold, peak_threshold = np.quantile(
            np.abs(np_waveform), [0.5, 0.95]
        )
        ratio = silence_threshold / peak_threshold
        if ratio > 0.15:
            n_speakers = 2
        else:
            n_speakers = 1
        return {"value": n_speakers, "probability": 0.85}

elif _diarization_service == "pyannote":
    _diarization_pipeline = pyannote.audio.Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        # TODO: Token seems useless (any string works). Leaving it here just for safety.
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
    )
    if _diarization_pipeline is None:
        raise ValueError("pipeline not initialized")
    # send pipeline to GPU (when available)
    if _diarization_pipeline is not None and run_opts.get("device") == "cuda":
        _diarization_pipeline.to(torch.device("cuda"))

    def _get_speaker_count(audio: _core.Audio) -> dict:
        # NOTE: Denoising is not really working. Commenting out for now.
        # sources = _denoiser.separate_batch(audio.waveform)
        # clean_signal = sources[:, :, 0]
        # audio = _core.Audio(waveform=clean_signal, sample_rate=audio.sample_rate)
        diarization = _diarization_pipeline(
            audio.to_dict(), min_speakers=0, max_speakers=3
        )
        # TODO: We are throwing away a lot of info here...
        n_speakers = len(diarization.labels())
        score = 1 - (
            diarization.get_overlap().duration() / diarization.get_timeline().duration()
        )
        return {"value": n_speakers, "probability": score}

elif _diarization_service != "":
    # TODO: There is an alternative diarizer here, but it is not as easy to use as pyannote.
    # Integration looks more cumbersome.
    # https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb
    raise ValueError(
        f'Unsupported diarization service "{_diarization_service}". '
        f"Use `local` or `pyannote`"
    )


def get_speaker_count(audio: _core.Audio) -> dict:
    return _get_speaker_count(audio)


##############
# Embeddings
##############
_speaker_encoder = classifiers.EncoderClassifier.from_hparams(
    # source="speechbrain/spkrec-xvect-voxceleb",
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/app/speechbrain_models",
    run_opts=run_opts,
)

SIMILARITY_THRESHOLD=0.25
cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

def compute_embedding(audio: _core.Audio) -> torch.Tensor:
    return _speaker_encoder.encode_batch(audio.waveform)

def similarity(ref_embedding: torch.Tensor, audio_embedding: torch.Tensor) -> float:
    s = cosine_similarity(audio_embedding, ref_embedding)
    return float(s[0])

def embedding_to_base64(audio_embedding: torch.Tensor) -> str:
    # TODO: Skipping the next check for now. Needs resampling
    # if audio.sample_rate != 16000:
    #     # See https://huggingface.co/speechbrain/spkrec-xvect-voxceleb
    #     raise ValueError('Embedding models only works at 16kHz, '
    #                      f'not {audio.sample_rate / 1000:.03f}kHz')
    buffer = BytesIO()
    torch.save(audio_embedding, buffer)
    b64 = base64.standard_b64encode(buffer.getvalue()).decode()
    return b64

def embedding_from_base64(ref_embedding_base64: str) -> torch.Tensor | None:
    if ref_embedding_base64 is None or ref_embedding_base64 == "":
        return None
    buffer = BytesIO(base64.standard_b64decode(ref_embedding_base64))
    return torch.load(buffer, weights_only=True)

def similarity_matrix(embeddings: list):
    vector = [embedding_from_base64(e) for e in embeddings]
    n = len(vector)
    matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j]  = 1.0
                continue
            matrix[i][j] = similarity(vector[i], vector[j])
    
    return matrix

##########################
# Emotion classification
##########################
_emotion_classifier = interfaces.foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    savedir="/app/speechbrain_models/emotion-recognition-wav2vec2-IEMOCAP",
    run_opts=run_opts,
)


def get_emotion(audio: _core.Audio) -> dict:
    _, score, _, label = _emotion_classifier.classify_batch(audio.waveform)
    return {"label": label[0], "score": score[0].tolist()}


##########################
# Language classification
##########################
_language_classifier = classifiers.EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-commonlanguage_ecapa",
    savedir="/app/speechbrain_models/lang-id-commonlanguage_ecapa",
    run_opts=run_opts,
)


def get_language(audio: _core.Audio) -> dict:
    _, score, _, label = _language_classifier.classify_batch(audio.waveform)
    return {"label": label[0], "score": score[0].tolist()}


#########################
# Separation
#########################
_n_speaker_to_separator = {
    2: separation.SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir="/app/speechbrain_models/sepformer-wsj02mix",
    ),
    3: separation.SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj03mix",
        savedir="/app/speechbrain_models/sepformer-wsj03mix",
    ),
}


def separate(audio: _core.Audio, n_speakers: int) -> str:
    separator = _n_speaker_to_separator[n_speakers]
    separated = separator.separate_batch(audio.waveform)
    return base64.standard_b64encode(pickle.dumps(separated)).decode()
