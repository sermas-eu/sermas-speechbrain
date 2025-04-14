import base64
import os
import pickle
from io import BytesIO
import numpy as np
import pyannote.audio
import torch
from speechbrain.inference import classifiers, interfaces, separation
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
            use_gpu = False

        if "USE_GPU" not in os.environ.keys():
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

        self._diarization_service = os.environ.get(
            "DIARIZATION_SERVICE", "local"
        ).lower()

        self.SIMILARITY_THRESHOLD = 0.25
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def get_speechbrain_cache_path(self, path: str) -> str:
        return os.path.join(self.speechbrain_cache_dir, path)

    def load_models(self):

        self.logger.info("Loading models")

        ################
        # Denoising
        ################
        self._denoiser = separation.SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr-enhancement",
            savedir=self.get_speechbrain_cache_path("sepformer-whamr-enhancement"),
        )

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

        #########################
        # Separation
        #########################
        self._n_speaker_to_separator = {
            2: separation.SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir=self.get_speechbrain_cache_path("sepformer-wsj02mix"),
            ),
            3: separation.SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj03mix",
                savedir=self.get_speechbrain_cache_path("sepformer-wsj03mix"),
            ),
        }

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

        ###############
        # Diarization
        ###############

        self.logger.info(f"Using {self._diarization_service} diarization")
        if self._diarization_service == "pyannote":

            pyannote_model = "pyannote/speaker-diarization-3.1"
            self.logger.info(f"Loading model {pyannote_model}")

            self._diarization_pipeline = pyannote.audio.Pipeline.from_pretrained(
                pyannote_model,
                # TODO: Token seems useless (any string works). Leaving it here just for safety.
                use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
            )
            if self._diarization_pipeline is None:
                self.logger.error("Failed to initialize pyannote pipeline")
                raise ValueError("pipeline not initialized")

            # send pipeline to GPU (when available)
            if (
                self._diarization_pipeline is not None
                and self.run_opts.get("device") == "cuda"
            ):
                self._diarization_pipeline.to(torch.device("cuda"))
        elif self._diarization_service == "local":
            pass
        else:
            # TODO: There is an alternative diarizer here, but it is not as easy to use as pyannote.
            # Integration looks more cumbersome.
            # https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb
            raise ValueError(
                f'Unsupported diarization service "{self._diarization_service}". '
                f"Use `local` or `pyannote`"
            )

    def get_speaker_count(self, audio: core.Audio) -> dict:
        if self._diarization_service == "local":
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
        else:
            # NOTE: Denoising is not really working. Commenting out for now.
            # sources = _denoiser.separate_batch(audio.waveform)
            # clean_signal = sources[:, :, 0]
            # audio = core.Audio(waveform=clean_signal, sample_rate=audio.sample_rate)
            diarization = self._diarization_pipeline(
                audio.to_dict(), min_speakers=0, max_speakers=3
            )
            # TODO: We are throwing away a lot of info here...
            n_speakers = len(diarization.labels())
            score = 1 - (
                diarization.get_overlap().duration()
                / diarization.get_timeline().duration()
            )
            return {"value": n_speakers, "probability": score}

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

    def similarity_matrix(self, embeddings: list) -> list[list[float]]:
        vector = [self.embedding_from_base64(e) for e in embeddings]
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

    def separate(self, audio: core.Audio, n_speakers: int) -> str:
        separator = self._n_speaker_to_separator[n_speakers]
        separated = separator.separate_batch(audio.waveform)
        return base64.standard_b64encode(pickle.dumps(separated)).decode()
