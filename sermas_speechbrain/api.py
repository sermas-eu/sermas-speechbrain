import logging
import json
import flask_cors
from flask import Flask, request
from werkzeug import exceptions
from waitress import serve
from sermas_speechbrain import config, core, speech_models

config.load_env()

models = speech_models.SpeechModelWrapper()
models.load_models()

logger = logging.getLogger(__name__)
logger.info("Starting speech detection API")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
flask_cors.CORS(app)


def _to_dict(score, label) -> dict:
    return {"label": label[0], "score": score[0].tolist()}


@app.before_request
def log_request_info():
    app.logger.info("%s %s", request.method, request.path)
    app.logger.debug("Headers: %s", request.path)


@app.route("/", methods=["POST"])
def all():
    logging.warning("DEPRECATED Classify request.")

    audio = _get_audio_from_request()

    try:
        return {
            "language": models.get_language(audio),
            "emotion": models.get_emotion(audio),
            # 'speakerId': speakerid(audio),  # This cannot be done without a trained model
        }
    except Exception as e:
        return f"Error processing classify request {e}", 500


def _get_audio_from_request():
    if "file" not in request.files:
        raise exceptions.BadRequest("No file attached")
    audiofile = request.files["file"]
    if not audiofile.filename:
        raise exceptions.BadRequest("No file selected")
    try:
        audio = core.Audio.from_file(audiofile)
    except Exception as e:  # TODO: Exception too broad
        raise exceptions.BadRequest(f"Error reading file {e}")
    return audio


@app.route("/", methods=["GET"])
def hello_world():
    return "Hello speechbrain!", 200


@app.errorhandler(404)
def page_not_found(e):
    app.logger.error("No handler found for %s", request.path)
    return {
        "error": True,
        "message": "Could not find a handler for " + request.path,
    }, 404


@app.route("/separate", methods=["POST"])
def separate():

    audio = _get_audio_from_request()

    n_speakers = request.form.get("n_speakers")
    if n_speakers:
        n_speakers = int(n_speakers)
        speaker_count = {"value": n_speakers, "probability": 1.0}
    else:
        speaker_count = models.get_speaker_count(audio)
        n_speakers = speaker_count["value"]

    if n_speakers < 2:
        return {"speakerCount": speaker_count}, 200

    if n_speakers > 3:
        raise exceptions.BadRequest(f"{n_speakers} speakers detected. Max is 3")

    signals = models.separate(audio, n_speakers)
    return {"signals": signals, "speakerCount": speaker_count}, 200


@app.route("/count_speakers", methods=["POST"])
def count_speakers():
    audio = _get_audio_from_request()
    speaker_count = models.get_speaker_count(audio)
    return {"speakerCount": speaker_count}, 200


@app.route("/verify_speakers", methods=["POST"])
def verify_speakers():
    try:
        audio = _get_audio_from_request()
        audio_embedding = models.compute_embedding(audio)

        embeddings = request.form.get("embeddings")
        ref_embeddings_base64 = json.loads(embeddings)
        res = []
        for emb in ref_embeddings_base64:
            if emb == "":
                res.append(None)
                continue
            ref_embeddings = models.embedding_from_base64(emb)
            res.append(models.similarity(ref_embeddings, audio_embedding))

        return {
            "similarities": res,
            "embeddings": models.embedding_to_base64(audio_embedding),
        }, 200
    except Exception as e:
        print(e)
        return f"Error processing verify request {e}", 500


@app.route("/similarity_matrix", methods=["POST"])
def similarity_matrix():
    try:
        embeddings = request.form.get("embeddings")
        matrix = models.similarity_matrix(json.loads(embeddings))
        return {"similarity_matrix": matrix}, 200
    except Exception as e:
        return f"Error processing verify request {e}", 500


@app.route("/create_embeddings", methods=["POST"])
def create_embeddings():

    audio = _get_audio_from_request()

    try:
        audio_embedding = models.compute_embedding(audio)
        return {
            "embeddings": models.embedding_to_base64(
                audio_embedding
            ),  # TODO: Implement resample
        }
    except Exception as e:
        return f"Error processing create embedding request {e}", 500


if __name__ == "__main__":
    serve(app, listen="*:5011")
