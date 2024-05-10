
import datetime
import json
import logging
import io

import numpy as np
import torch
import torchaudio
from flask import Flask, abort, render_template, request
from flask_cors import CORS
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.inference.interfaces import foreign_class
from waitress import serve

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
CORS(app)

logging.basicConfig(level=logging.INFO)

run_opts = {
    # enable CUDA
    # "device": "cuda",

    # "data_parallel_count": -1,
    # "data_parallel_backend": False,
    # "distributed_launch": False,
    # "distributed_backend": "nccl",
    # "jit_module_keys": None
}

speaker_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir='models/spkrec-xvect-voxceleb',
    run_opts=run_opts
)

emotion_calssifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
    savedir='models/emotion-recognition-wav2vec2-IEMOCAP',
    run_opts=run_opts
)

lang_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-commonlanguage_ecapa",
    savedir="models/lang-id-commonlanguage_ecapa",
    run_opts=run_opts
)

def load_data(filestore):
    try:
        filestore.seek(0)
        raw = np.fromstring(filestore.read(), dtype=np.uint8)
        audio = torch.from_numpy(raw)
        return audio
    except Exception as e:
        logging.error(f"Failed to load wav {e}")
        return None


def json_result(score, label):
    return {'label': label[0], 'score': score[0].tolist()}


def language(data):
    output_probs, score, index, label = lang_classifier.classify_batch(data)
    logging.info(f"Language label: {label}, score: {score}")
    return json_result(score, label)


def speakerid(data):
    output_probs, score, index, label = speaker_classifier.classify_batch(data)
    logging.info(f"SpeakerID label: {label}, score: {score}")
    return json_result(score, label)


def emotion(data):
    output_probs, score, index, label = emotion_calssifier.classify_batch(data)
    logging.info(f"Emotion label: {label}, score: {score}")
    return json_result(score, label)


# @app.route('/language', methods = ['POST'])
# def language_request():
#     logging.info("Language classification request")
#     data = request.get_data()
#     audio = load_data(data)
#     return speakerid(audio)

# @app.route('/speakerid', methods = ['POST'])
# def speakerid_request():
#     logging.info("SpeakerID classification request")
#     data = request.get_data()
#     audio = load_data(data)
#     return speakerid(audio)

# @app.route('/emotion', methods = ['POST'])
# def emotion_request():
#     logging.info("Emotion classification request")
#     data = request.get_data()
#     audio = load_data(data)
#     return emotion(audio)


@app.before_request
def log_request_info():
    now = datetime.datetime.now().isoformat()
    print("[" + now + "] INFO: handling " + request.method + " request for " + request.path)
    app.logger.debug('Headers: %s', request.path)


@app.route('/', methods=['POST'])
def all():
    logging.info("Classify request")
    try:
        if 'file' not in request.files:
            logging.warning("No file attached")
            return 'No file attached', 400

        file = request.files['file']
        if file.filename == '':
            logging.warning("No file selected")
            return 'No file selected', 400

        filestore = io.BytesIO()
        file.save(filestore)
        audio = load_data(filestore)

        if audio is None:
            logging.warning("Cannot parse content")
            return 'Cannot parse content', 400

        return json.dumps({
            # "language": language(audio),
            "emotion": emotion(audio),
            "speakerId": speakerid(audio)
        })

    except Exception as e:
        logging.info(f"Error {e}")
        return 'Error processing request', 500


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello speechbrain!', 200


@app.errorhandler(404)
def page_not_found(e):
    now = datetime.datetime.now().isoformat()
    print("[" + now + "] ERROR: could not find a handler for " + request.path)
    return { 'error': True, 'message': 'Could not find a handler for ' + request.path }, 404


def test():
    signal = load_data('/data/test.wav')
    res = json.dumps({"emotion": emotion(signal),
                     "speakerId": speakerid(signal)})
    logging.info(f"Test result: {res}")


if __name__ == '__main__':
    # test()
    serve(app, listen='*:5011')
