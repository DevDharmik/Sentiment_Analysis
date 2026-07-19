import unittest

from xquik_export import ExportFormatError, load_xquik_rows


class XquikExportTests(unittest.TestCase):
    def test_loads_nested_json_rows(self):
        payload = b'{"comments":[{"tweet":"Great video","likes":7,"username":"ana"}]}'

        rows = load_xquik_rows(payload)

        self.assertEqual(
            rows,
            [
                {
                    "author": "ana",
                    "text": "Great video",
                    "likes": 7,
                    "published": "",
                }
            ],
        )

    def test_loads_jsonl_content_rows(self):
        payload = b'{"content":"Helpful"}\n{"full_text":"Needs work","favorite_count":"2"}'

        rows = load_xquik_rows(payload)

        self.assertEqual(rows[0]["text"], "Helpful")
        self.assertEqual(rows[1]["likes"], 2)

    def test_loads_csv_body_rows(self):
        payload = b"body,created_at,screen_name\nClean chart,2026-01-02,bob\n"

        rows = load_xquik_rows(payload)

        self.assertEqual(
            rows,
            [
                {
                    "author": "bob",
                    "text": "Clean chart",
                    "likes": 0,
                    "published": "2026-01-02",
                }
            ],
        )

    def test_skips_blank_text(self):
        payload = b'[{"text":" "},{"body":"Useful export"}]'

        rows = load_xquik_rows(payload)

        self.assertEqual([row["text"] for row in rows], ["Useful export"])

    def test_rejects_malformed_json(self):
        with self.assertRaisesRegex(ExportFormatError, "not valid"):
            load_xquik_rows(b'{"tweets":')

    def test_rejects_non_utf8_exports(self):
        with self.assertRaisesRegex(ExportFormatError, "UTF-8"):
            load_xquik_rows(b"\xff\xfe")


if __name__ == "__main__":
    unittest.main()
