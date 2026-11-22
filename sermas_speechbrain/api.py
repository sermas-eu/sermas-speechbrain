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


@app.route("/", methods=["POST"])
def classify():

    audio = _get_audio_from_request()

    return {
        "language": models.get_language(audio),
        "emotion": models.get_emotion(audio),
    }


def _get_audio_from_request():
    if "file" not in request.files:
        raise exceptions.BadRequest("No file attached")
    audiofile = request.files["file"]
    if not audiofile.filename:
        raise exceptions.BadRequest("No file selected")
    try:
        audio = core.Audio.from_file(audiofile)
    except Exception as e:  # any decoding failure is the caller's input
        raise exceptions.BadRequest(f"Error reading file {e}")
    return audio


@app.route("/", methods=["GET"])
def hello_world():
    return "Hello speechbrain!", 200


@app.errorhandler(exceptions.InternalServerError)
def internal_error(e):
    logger.exception("Request failed: %s", request.path)
    return {"error": True, "message": "Internal error"}, 500


@app.errorhandler(404)
def page_not_found(e):
    app.logger.error("No handler found for %s", request.path)
    return {
        "error": True,
        "message": "Could not find a handler for " + request.path,
    }, 404


def _get_embeddings_from_request() -> list:
    """The `embeddings` form field: a JSON list of base64 embeddings ("" for none)."""
    raw = request.form.get("embeddings")
    if raw is None:
        raise exceptions.BadRequest("No embeddings given")
    try:
        embeddings = json.loads(raw)
    except ValueError as e:
        raise exceptions.BadRequest(f"embeddings is not JSON: {e}")
    if not isinstance(embeddings, list) or not all(
        isinstance(e, str) for e in embeddings
    ):
        raise exceptions.BadRequest("embeddings must be a list of strings")
    return embeddings


def _decode(embedding: str):
    try:
        return models.embedding_from_base64(embedding)
    except Exception as e:
        raise exceptions.BadRequest(f"Invalid embedding: {e}")


@app.route("/verify_speakers", methods=["POST"])
def verify_speakers():
    audio = _get_audio_from_request()
    refs = [_decode(e) if e else None for e in _get_embeddings_from_request()]
    audio_embedding = models.compute_embedding(audio)
    return {
        "similarities": [
            None if ref is None else models.similarity(ref, audio_embedding)
            for ref in refs
        ],
        "embeddings": models.embedding_to_base64(audio_embedding),
    }, 200


@app.route("/similarity_matrix", methods=["POST"])
def similarity_matrix():
    embeddings = _get_embeddings_from_request()
    if not all(embeddings):
        raise exceptions.BadRequest("empty embedding in the list")
    vectors = [_decode(e) for e in embeddings]
    return {"similarity_matrix": models.similarity_matrix(vectors)}, 200


@app.route("/create_embeddings", methods=["POST"])
def create_embeddings():
    audio = _get_audio_from_request()
    return {"embeddings": models.embedding_to_base64(models.compute_embedding(audio))}


if __name__ == "__main__":
    serve(app, listen="*:5011")
