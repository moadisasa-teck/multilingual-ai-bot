import unittest

from app.chatbot import Chatbot


class DeterministicFormatTests(unittest.TestCase):
    def setUp(self):
        self.bot = Chatbot.__new__(Chatbot)

    def test_single_answer_returns_plain_text(self):
        ranked_hits = [
            {
                "score": 0.9,
                "meta": {
                    "answer": "You need a completed application form. (DEMO)",
                    "sector": "passport",
                    "source_file": "passport_en.csv",
                },
            }
        ]
        answer, _, _, source_file = self.bot._compose_extractive_answer(ranked_hits, "en")
        self.assertEqual(answer, "You need a completed application form.")
        self.assertEqual(source_file, "passport_en.csv")

    def test_concise_multi_answer_returns_paragraph(self):
        ranked_hits = [
            {
                "score": 0.93,
                "meta": {
                    "answer": "You need a completed application form. (DEMO)",
                    "sector": "passport",
                    "source_file": "a.csv",
                },
            },
            {
                "score": 0.91,
                "meta": {
                    "answer": "Normal processing takes several weeks. (DEMO)",
                    "sector": "passport",
                    "source_file": "b.csv",
                },
            },
        ]
        answer, _, _, source_file = self.bot._compose_extractive_answer(ranked_hits, "en")
        self.assertIn("Here is what I found:", answer)
        self.assertNotIn("\n- ", answer)
        self.assertEqual(source_file, "multi_concise")

    def test_long_multi_answer_returns_intro_plus_bullets(self):
        ranked_hits = [
            {
                "score": 0.95,
                "meta": {
                    "answer": "You need a completed application form, national ID or birth certificate, photos, and proof of payment. (DEMO)",
                    "sector": "passport",
                    "source_file": "a.csv",
                },
            },
            {
                "score": 0.94,
                "meta": {
                    "answer": "Normal processing takes several weeks while expedited services are faster depending on workload and requirements. (DEMO)",
                    "sector": "passport",
                    "source_file": "b.csv",
                },
            },
            {
                "score": 0.93,
                "meta": {
                    "answer": "Report loss immediately and apply for replacement with a police report and supporting documents. (DEMO)",
                    "sector": "passport",
                    "source_file": "c.csv",
                },
            },
        ]
        answer, _, _, source_file = self.bot._compose_extractive_answer(ranked_hits, "en")
        self.assertIn("I found the following details:", answer)
        self.assertIn("\n- ", answer)
        self.assertEqual(source_file, "multi_source")


if __name__ == "__main__":
    unittest.main()
