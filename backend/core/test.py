from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status


class GeminiTranslateAPITest(APITestCase):
    def test_translate_success(self):
        """
        Test POST /api/gemini/translate/
        """
        url = "/api/gemini/translate/"
        payload = {
            "text": "日本語の勉強は楽しいです"
        }

        response = self.client.post(
            url,
            payload,
            format="json"
        )

        print("\n=== RESPONSE DATA ===")
        print(response.data)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("japanese", response.data)
        self.assertIn("vietnamese", response.data)
