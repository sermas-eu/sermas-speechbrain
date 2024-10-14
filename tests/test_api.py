import unittest

from sermas_speechbrain import api


class APITests(unittest.TestCase):
    def setUp(self):
        self.ctx = api.app.app_context()
        self.ctx.push()
        self.client = api.app.test_client()

    def tearDown(self):
        self.ctx.pop()

    def test_hello_speechbrain(self):
        response = self.client.get('/')
        self.assertEqual(200, response.status_code)
        self.assertEqual('Hello speechbrain!', response.get_data(as_text=True))


if __name__ == '__main__':
    unittest.main()
