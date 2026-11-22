"""The endpoints sermas-api calls: classify (`POST /`) and the identity tracker's
`/create_embeddings`, `/verify_speakers`, `/similarity_matrix`.

Checks hold across CPU/GPU and library versions: labels, shapes and ranges, and the
ordering of similarities (same speaker above other speakers), not exact scores.
"""

import base64
import io
import json
import pathlib
import unittest

from sermas_speechbrain import api

DATA = pathlib.Path(__file__).parent / "data"
SUFFIX = "-251-136532-0015_1993-147965-0008_7976-110124-0014.wav"
S1_NOISE = DATA / f"02-s1_with_noise{SUFFIX}"
NOISE = DATA / f"03-noise_only{SUFFIX}"
S1 = DATA / f"04-s1_only{SUFFIX}"
S2 = DATA / f"05-s2_only{SUFFIX}"
S3 = DATA / f"06-s3_only{SUFFIX}"

# the IEMOCAP model's classes
EMOTIONS = {"neu", "ang", "hap", "sad"}


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = api.app.test_client()
        cls._embeddings = {}

    def post(self, endpoint, audio=None, **form):
        data = dict(form)
        if audio is not None:
            data["file"] = (open(audio, "rb"), audio.name)
        try:
            return self.client.post(
                endpoint, data=data, content_type="multipart/form-data"
            )
        finally:
            if audio is not None:
                data["file"][0].close()

    def embedding(self, audio) -> str:
        if audio not in self._embeddings:
            res = self.post("/create_embeddings", audio)
            self.assertEqual(200, res.status_code, res.get_data(as_text=True))
            self._embeddings[audio] = res.json["embeddings"]
        return self._embeddings[audio]

    def matrix(self, embeddings):
        res = self.post("/similarity_matrix", embeddings=json.dumps(embeddings))
        self.assertEqual(200, res.status_code, res.get_data(as_text=True))
        return res.json["similarity_matrix"]

    def test_hello(self):
        res = self.client.get("/")
        self.assertEqual(200, res.status_code)
        self.assertEqual("Hello speechbrain!", res.get_data(as_text=True))

    # classify

    def test_classify_returns_language_and_emotion(self):
        for audio in (S1, S2, S3, NOISE):
            with self.subTest(audio=audio.name[:2]):
                res = self.post("/", audio)
                self.assertEqual(200, res.status_code, res.get_data(as_text=True))
                emotion, language = res.json["emotion"], res.json["language"]
                self.assertIn(emotion["label"], EMOTIONS)
                self.assertGreaterEqual(emotion["score"], 0.0)
                self.assertLessEqual(emotion["score"], 1.0 + 1e-6)
                self.assertIsInstance(language["label"], str)
                self.assertTrue(language["label"])
                self.assertIsInstance(language["score"], float)

    def test_classify_without_file_is_400(self):
        self.assertEqual(400, self.post("/").status_code)

    def test_classify_with_non_audio_is_400(self):
        res = self.client.post(
            "/",
            data={"file": (io.BytesIO(b"not audio"), "x.wav")},
            content_type="multipart/form-data",
        )
        self.assertEqual(400, res.status_code)

    # embeddings

    def test_create_embeddings_returns_base64(self):
        emb = self.embedding(S1)
        self.assertIsInstance(emb, str)
        self.assertGreater(len(base64.standard_b64decode(emb)), 0)

    def test_embeddings_are_stable(self):
        again = self.post("/create_embeddings", S1).json["embeddings"]
        m = self.matrix([self.embedding(S1), again])
        self.assertAlmostEqual(1.0, m[0][1], places=4)

    # similarity matrix

    def test_similarity_matrix_shape(self):
        audios = (S1, S2, S3)
        m = self.matrix([self.embedding(a) for a in audios])
        self.assertEqual(3, len(m))
        for i in range(3):
            self.assertEqual(3, len(m[i]))
            self.assertAlmostEqual(1.0, m[i][i], places=4)
            for j in range(3):
                self.assertAlmostEqual(m[i][j], m[j][i], places=6)
                self.assertGreaterEqual(m[i][j], -1.0 - 1e-6)
                self.assertLessEqual(m[i][j], 1.0 + 1e-6)

    def test_same_speaker_is_closer_than_others(self):
        m = self.matrix([self.embedding(a) for a in (S1, S1_NOISE, S2, S3)])
        same = m[0][1]
        self.assertGreater(same, m[0][2])
        self.assertGreater(same, m[0][3])

    # verify speakers

    def test_verify_speakers(self):
        refs = [self.embedding(S1), self.embedding(S2), ""]
        res = self.post("/verify_speakers", S1, embeddings=json.dumps(refs))
        self.assertEqual(200, res.status_code, res.get_data(as_text=True))
        sims = res.json["similarities"]
        self.assertEqual(3, len(sims))
        self.assertAlmostEqual(1.0, sims[0], places=4)
        self.assertLess(sims[1], sims[0])
        self.assertIsNone(sims[2])
        self.assertIsInstance(res.json["embeddings"], str)

    def test_verify_speakers_with_bad_embeddings_is_400(self):
        res = self.post("/verify_speakers", S1, embeddings="not json")
        self.assertEqual(400, res.status_code)

    def test_similarity_matrix_with_bad_embeddings_is_400(self):
        self.assertEqual(400, self.post("/similarity_matrix").status_code)


if __name__ == "__main__":
    unittest.main()
