import base64
import os
from io import BytesIO
import torch
from speechbrain.inference import classifiers, interfaces
import logging
from sermas_speechbrain import core


class SpeechModelWrapper:

    def __init__(self):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

        # TODO: This can be a lot cleaner
        self.run_opts = {
            # enable CUDA
            "device": "cuda",
            "data_parallel_count": -1,
            "data_parallel_backend": False,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "jit_module_keys": None,
        }

        use_gpu = True
        # defaults to GPU
        if os.environ.get("USE_GPU") == "0":
            self.logger.info("GPU disabled")
            use_gpu = False

        if "USE_GPU" not in os.environ.keys():
            self.logger.info("GPU requested")
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                self.logger.warning("GPU is not available, reverting to CPU")
                use_gpu = False

        if use_gpu:
            self.logger.info("Using GPU")
        else:
            self.logger.info("Using CPU")
            self.run_opts = {}

        # in docker defaults  to /cache/speechbrain
        self.speechbrain_cache_dir = os.environ.get(
            "SPEECHBRAIN_CACHE_DIR", "./speechbrain_models"
        )

        self.SIMILARITY_THRESHOLD = 0.25
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def get_speechbrain_cache_path(self, path: str) -> str:
        return os.path.join(self.speechbrain_cache_dir, path)

    def load_models(self):

        self.logger.info("Loading models")

        ##############
        # Embeddings
        ##############
        self._speaker_encoder = classifiers.EncoderClassifier.from_hparams(
            # source="speechbrain/spkrec-xvect-voxceleb",
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=self.get_speechbrain_cache_path("spkrec-ecapa-voxceleb"),
            run_opts=self.run_opts,
        )

        ##########################
        # Language classification
        ##########################
        self._language_classifier = classifiers.EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-commonlanguage_ecapa",
            savedir=self.get_speechbrain_cache_path("lang-id-commonlanguage_ecapa"),
            run_opts=self.run_opts,
        )

        ##########################
        # Emotion classification
        ##########################
        self._emotion_classifier = interfaces.foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=self.get_speechbrain_cache_path(
                "emotion-recognition-wav2vec2-IEMOCAP"
            ),
            run_opts=self.run_opts,
        )

    def compute_embedding(self, audio: core.Audio) -> torch.Tensor:
        return self._speaker_encoder.encode_batch(audio.waveform)

    def similarity(
        self, ref_embedding: torch.Tensor, audio_embedding: torch.Tensor
    ) -> float:
        s = self.cosine_similarity(audio_embedding, ref_embedding)
        return float(s[0])

    def embedding_to_base64(self, audio_embedding: torch.Tensor) -> str:
        # TODO: Skipping the next check for now. Needs resampling
        # if audio.sample_rate != 16000:
        #     # See https://huggingface.co/speechbrain/spkrec-xvect-voxceleb
        #     raise ValueError('Embedding models only works at 16kHz, '
        #                      f'not {audio.sample_rate / 1000:.03f}kHz')
        buffer = BytesIO()
        torch.save(audio_embedding, buffer)
        b64 = base64.standard_b64encode(buffer.getvalue()).decode()
        return b64

    def embedding_from_base64(self, ref_embedding_base64: str) -> torch.Tensor | None:
        if not ref_embedding_base64:
            return None
        buffer = BytesIO(base64.standard_b64decode(ref_embedding_base64))
        return torch.load(buffer, weights_only=True)

    def similarity_matrix(self, vector: list[torch.Tensor]) -> list[list[float]]:
        n = len(vector)
        matrix = [[1.0] * n for _ in range(n)]  # n x n matrix of ones

        for i in range(n):
            for j in range(i):  # computing only lower triangle
                matrix[i][j] = self.similarity(vector[i], vector[j])
                matrix[j][i] = matrix[i][j]  # matrix is symmetric

        return matrix

    def get_emotion(self, audio: core.Audio) -> dict:
        _, score, _, label = self._emotion_classifier.classify_batch(audio.waveform)
        return {"label": label[0], "score": score[0].tolist()}

    def get_language(self, audio: core.Audio) -> dict:
        _, score, _, label = self._language_classifier.classify_batch(audio.waveform)
        return {"label": label[0], "score": score[0].tolist()}
