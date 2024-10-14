import logging

import flask_cors
from flask import Flask, request
from waitress import serve

from sermas_speechbrain import _core, _models

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
flask_cors.CORS(app)

logging.basicConfig(level=logging.INFO)

def _to_dict(score, label) -> dict:
    return {'label': label[0], 'score': score[0].tolist()}


# def language(data):
#     output_probs, score, index, label = lang_classifier.classify_batch(data)
#     logging.info(f'Language label: {label}, score: {score}')
#     return json_result(score, label)
#
#
# def speakerid(data):
#     output_probs, score, index, label = speaker_classifier.classify_batch(data)
#     logging.info(f'SpeakerID label: {label}, score: {score}')
#     return json_result(score, label)
#
#
# def emotion(data):
#     output_probs, score, index, label = emotion_classifier.classify_batch(data)
#     logging.info(f'Emotion label: {label}, score: {score}')
#     return json_result(score, label)





@app.before_request
def log_request_info():
    app.logger.info('%s %s', request.method, request.path)
    app.logger.debug('Headers: %s', request.path)


@app.route('/', methods=['POST'])
def all():
    logging.warning('DEPRECATED Classify request.')

    if 'file' not in request.files:
        return 'No file attached', 400

    audiofile = request.files['file']
    if not audiofile.filename:
        return 'No file selected', 400
    try:
        audio = _core.Audio.from_file(audiofile)
    except Exception as e:  # TODO: Exception too broad
        return f'Error reading file {e}', 400

    try:
        return {
            'language': _models.get_language(audio),
            'emotion': _models.get_emotion(audio),
            # 'speakerId': speakerid(audio),  # This cannot be done without a trained model
            'embeddings': _models.get_embeddings(audio)
        }
    except Exception as e:
        return 'Error processing request', 500


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello speechbrain!', 200


@app.errorhandler(404)
def page_not_found(e):
    app.logger.error('No handler found for %s', request.path)
    return { 'error': True, 'message': 'Could not find a handler for ' + request.path }, 404


@app.route('/separate', methods=['POST'])
def separate():
    # # apply pretrained pipeline
    # diarization = pipeline('audio.wav')
    #
    # # print the result
    # for turn, _, speaker in diarization.itertracks(yield_label=True):
    #     print(f'start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}')
    return 'Hello speechbrain!', 200


if __name__ == '__main__':
    serve(app, listen='*:5011')
