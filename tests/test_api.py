import base64
import hashlib
import pathlib
import pickle
import unittest
from typing import Union, Optional

import torch
import werkzeug.test

from sermas_speechbrain import api

_current_folder = pathlib.Path(__file__).parent
_data_folder = _current_folder / 'data'


class TestFiles:
    THREE_SPEAKERS_W_NOISE = _data_folder / '01-s1s2s3_with_noise-251-136532-0015_1993-147965-0008_7976-110124-0014.wav'
    SPEAKER1_W_NOISE = _data_folder / '02-s1_with_noise-251-136532-0015_1993-147965-0008_7976-110124-0014.wav'
    NOISE = _data_folder / '03-noise_only-251-136532-0015_1993-147965-0008_7976-110124-0014.wav'
    SPEAKER1 = _data_folder / '04-s1_only-251-136532-0015_1993-147965-0008_7976-110124-0014.wav'
    SPEAKER2 = _data_folder / '05-s2_only-251-136532-0015_1993-147965-0008_7976-110124-0014.wav'
    SPEAKER3 = _data_folder / '06-s3_only-251-136532-0015_1993-147965-0008_7976-110124-0014.wav'


class APITests(unittest.TestCase):
    def setUp(self):
        self.context = api.app.app_context()
        self.context.push()
        self.client = api.app.test_client()

    def tearDown(self):
        self.context.pop()

    def _send_file(
            self,
            endpoint: str,
            audiofile: Union[str, pathlib.Path],
            data: Optional[dict] = None
    ) -> werkzeug.test.TestResponse:
        data = data or {}
        with open(audiofile, 'rb') as f:
            data['file'] = (f, str(audiofile))
            return self.client.post(
                endpoint,
                data=data,
                follow_redirects=True,
                content_type='multipart/form-data'
            )

    def test_hello_speechbrain(self):
        response = self.client.get('/')
        self.assertEqual(200, response.status_code)
        self.assertEqual('Hello speechbrain!', response.get_data(as_text=True))

    def test_root_endpoint(self):
        # NOTE: This endpoint is deprecated. Testing just for backward compatibility
        response = self._send_file('/', TestFiles.SPEAKER1)
        self.assertEqual(200, response.status_code)
        content = response.json
        # TODO: The encoding process is not deterministic. It can not be tested this way
        # embeddings_hash = hashlib.sha256(content['embeddings'].encode()).hexdigest()
        # expected_embeddings_hash = '8542209aac05540b5604e9e7e7a60e5061f2b653e4592e86624d132570c1d08f'
        # self.assertEqual(expected_embeddings_hash, embeddings_hash)
        self.assertDictEqual({'label': 'ang', 'score': 1.0},
                             content['emotion'])
        self.assertDictEqual({'label': 'Chinese_Hongkong', 'score': 0.36074525117874146},
                             content['language'])

    def test_one_speaker(self):
        signals = [
            TestFiles.SPEAKER1,
            TestFiles.SPEAKER2,
            TestFiles.SPEAKER3,
            # TestFiles.SPEAKER1_W_NOISE  # TODO: This finds 2 speakers, not one
        ]
        for s in signals:
            with self.subTest(signal=s):
                response = self._send_file('/separate', s)
                self.assertEqual(200, response.status_code)
                content = response.json
                self.assertDictEqual({'n_speakers': 1}, content)

    def test_three_speakers(self):
        response = self._send_file('/separate',
                                   TestFiles.THREE_SPEAKERS_W_NOISE)
        self.assertEqual(200, response.status_code)
        content = response.json
        self.assertEqual(3, content['n_speakers'])
        # TODO: Implement separation check


    def test_three_speakers_with_n_speakers(self):
        response = self._send_file('/separate',
                                   TestFiles.THREE_SPEAKERS_W_NOISE,
                                   data={'n_speakers': 3})
        self.assertEqual(200, response.status_code)
        content = response.json
        self.assertEqual(3, content['n_speakers'])
        tensor_bytes = base64.standard_b64decode(content['signals'].encode())
        tensor = pickle.loads(tensor_bytes)
        self.assertIsInstance(tensor, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
