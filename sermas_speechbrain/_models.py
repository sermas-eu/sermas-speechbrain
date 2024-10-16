import pickle
import base64
import torch
from speechbrain.inference import classifiers, interfaces, separation
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
    diarization = _diarization_pipeline(audio.to_dict(),
                                        min_speakers=0,
                                        max_speakers=3)
    # TODO: We are throwing away a lot of info here...
    return len(diarization.labels())


##############
# Embeddings
##############
_speaker_encoder = classifiers.EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir='speechbrain_models/spkrec-xvect-voxceleb',
    run_opts=run_opts
)

def get_embeddings(audio: _core.Audio) -> str:
    # TODO: Skipping the next check for now
    # if audio.sample_rate != 16000:
    #     # See https://huggingface.co/speechbrain/spkrec-xvect-voxceleb
    #     raise ValueError('Embedding models only works at 16kHz, '
    #                      f'not {audio.sample_rate / 1000:.03f}kHz')
    embeddings = _speaker_encoder.encode_batch(audio.waveform)
    return base64.standard_b64encode(pickle.dumps(embeddings)).decode()


##########################
# Emotion classification
##########################
_emotion_classifier = interfaces.foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    savedir='speechbrain_models/emotion-recognition-wav2vec2-IEMOCAP',
    run_opts=run_opts
)

def get_emotion(audio: _core.Audio) -> dict:
    _, score, _, label = _emotion_classifier.classify_batch(audio.waveform)
    return {'label': label[0], 'score': score[0].tolist()}


##########################
# Language classification
##########################
_language_classifier = classifiers.EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-commonlanguage_ecapa",
    savedir="speechbrain_models/lang-id-commonlanguage_ecapa",
    run_opts=run_opts
)

def get_language(audio: _core.Audio) -> dict:
    _, score, _, label = _language_classifier.classify_batch(audio.waveform)
    return {'label': label[0], 'score': score[0].tolist()}

#########################
# Separation
#########################
_n_speaker_to_separator = {
    2: separation.SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir='speechbrain_models/sepformer-wsj02mix'
    ),
    3: separation.SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj03mix",
        savedir='speechbrain_models/sepformer-wsj03mix'
    )
}

def separate(audio: _core.Audio, n_speakers: int) -> str:
    separator = _n_speaker_to_separator[n_speakers]
    separated = separator.separate_batch(audio.waveform)
    return base64.standard_b64encode(pickle.dumps(separated)).decode()
