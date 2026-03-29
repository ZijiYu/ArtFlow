from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.cot_layer.new_api_client import NewAPIClient


class NewAPIClientTests(unittest.TestCase):
    def test_reads_api_key_from_secret_file_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            secret_path = temp_path / "api.txt"
            secret_path.write_text("first-key\nsecond-key\n", encoding="utf-8")
            config_path = temp_path / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "api:",
                        "  key: \"\"",
                        f"  key_file: \"{secret_path}\"",
                        "  key_line: 2",
                        "  base_url: \"https://example.com/v1\"",
                        "  model: \"gpt-test\"",
                    ]
                ),
                encoding="utf-8",
            )

            client = NewAPIClient(config_path=str(config_path))

        self.assertEqual("second-key", client.api_key)
        self.assertEqual("https://example.com/v1", client.base_url)
        self.assertEqual("gpt-test", client.model)
        self.assertTrue(client.enabled)


if __name__ == "__main__":
    unittest.main()
