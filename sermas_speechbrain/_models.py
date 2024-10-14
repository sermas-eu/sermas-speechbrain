import pickle
import torch
from speechbrain.inference import classifiers, interfaces
import pyannote.audio


from sermas_speechbrain import _core


# TODO: This can be a lot cleaner
run_opts = {
    # enable CUDA
    # "device": "cuda",
    # "data_parallel_count": -1,
    # "data_parallel_backend": False,
    # "distributed_launch": False,
    # "distributed_backend": "nccl",
    # "jit_module_keys": None
}


###############
# Diarization
###############
_diarization_pipeline = pyannote.audio.Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE"
)

# send pipeline to GPU (when available)
if run_opts.get('device') == 'cuda':
    _diarization_pipeline.to(torch.device("cuda"))

def get_n_speakers(audio: _core.Audio) -> int:
    return 2


##############
# Embeddings
##############
_speaker_encoder = classifiers.EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir='models/spkrec-xvect-voxceleb',
    run_opts=run_opts
)

def get_embeddings(audio: _core.Audio) -> bytes:
    if audio.sample_rate != 16000:
        # See https://huggingface.co/speechbrain/spkrec-xvect-voxceleb
        raise ValueError('Embedding models only works at 16kHz, '
                         f'not {audio.sample_rate / 1000:.03f}kHz')
    embeddings = _speaker_encoder.encode_batch(audio.waveform)
    return pickle.dumps(embeddings)


##########################
# Emotion classification
##########################
_emotion_classifier = interfaces.foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    savedir='models/emotion-recognition-wav2vec2-IEMOCAP',
    run_opts=run_opts
)

def get_emotion(audio: _core.Audio) -> dict:
    return {}

##########################
# Language classification
##########################
_language_classifier = classifiers.EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-commonlanguage_ecapa",
    savedir="models/lang-id-commonlanguage_ecapa",
    run_opts=run_opts
)

def get_language(audio: _core.Audio) -> dict:
    return {}

#########################
# Separation
#########################
