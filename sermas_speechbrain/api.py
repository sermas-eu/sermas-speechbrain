import logging
import json
import flask_cors
from flask import Flask, request
from werkzeug import exceptions
from waitress import serve

from sermas_speechbrain import _core, _models

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
flask_cors.CORS(app)

logging.basicConfig(level=logging.INFO)

def _to_dict(score, label) -> dict:
    return {'label': label[0], 'score': score[0].tolist()}

# TODOs:
# - Implement DB connection


@app.before_request
def log_request_info():
    app.logger.info('%s %s', request.method, request.path)
    app.logger.debug('Headers: %s', request.path)


@app.route('/', methods=['POST'])
def all():
    logging.warning('DEPRECATED Classify request.')

    audio = _get_audio_from_request()

    try:
        audio_embedding = _models.compute_embedding(audio)
        return {
            'language': _models.get_language(audio),
            'emotion': _models.get_emotion(audio),
            # 'speakerId': speakerid(audio),  # This cannot be done without a trained model
            'embeddings': _models.embedding_to_base64(audio_embedding),  # TODO: Implement resample
        }
    except Exception as e:
        return f'Error processing classify request {e}', 500


def _get_audio_from_request():
    if 'file' not in request.files:
        raise exceptions.BadRequest('No file attached')
    audiofile = request.files['file']
    if not audiofile.filename:
        raise exceptions.BadRequest('No file selected')
    try:
        audio = _core.Audio.from_file(audiofile)
    except Exception as e:  # TODO: Exception too broad
        raise exceptions.BadRequest(f'Error reading file {e}')
    return audio


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello speechbrain!', 200


@app.errorhandler(404)
def page_not_found(e):
    app.logger.error('No handler found for %s', request.path)
    return { 'error': True, 'message': 'Could not find a handler for ' + request.path }, 404


@app.route('/separate', methods=['POST'])
def separate():

    audio = _get_audio_from_request()

    n_speakers = request.form.get('n_speakers')
    if n_speakers:
        n_speakers = int(n_speakers)
        speaker_count = {'value': n_speakers, 'probability': 1.0}
    else:
        speaker_count = _models.get_speaker_count(audio)
        n_speakers = speaker_count['value']

    if n_speakers < 2:
        return {'speakerCount': speaker_count}, 200

    if n_speakers > 3:
        raise exceptions.BadRequest(f'{n_speakers} speakers detected. Max is 3')

    signals = _models.separate(audio, n_speakers)
    return {'signals': signals, 'speakerCount': speaker_count}, 200

@app.route('/count_speakers', methods=['POST'])
def count_speakers():
    audio = _get_audio_from_request()
    speaker_count = _models.get_speaker_count(audio)
    return {'speakerCount': speaker_count}, 200

@app.route('/verify_speaker', methods=['POST'])
def verify_speaker():
    try:
        audio = _get_audio_from_request()
        ref_embeddings_base64 = request.form.get('embeddings')
        ref_embeddings = _models.embedding_from_base64(ref_embeddings_base64)
        audio_embedding = _models.compute_embedding(audio)
        similarity = _models.similarity(ref_embeddings, audio_embedding)
        return {'similarity': similarity}, 200
    except Exception as e:
        return f'Error processing verify request {e}', 500

@app.route('/similarity_matrix', methods=['POST'])
def similarity_matrix():
    try:
        embeddings = request.form.get('embeddings')
        matrix = _models.similarity_matrix(json.loads(embeddings))
        return {'similarity_matrix': matrix}, 200
    except Exception as e:
        return f'Error processing verify request {e}', 500

if __name__ == '__main__':
    serve(app, listen='*:5011')
