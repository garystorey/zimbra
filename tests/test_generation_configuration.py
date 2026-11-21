from pathlib import Path
import unittest

import config
import generate


class GenerationConfigurationTests(unittest.TestCase):
    def test_tts_defaults_match_voice_clone_tuning(self):
        self.assertEqual(config.TTS_TEMPERATURE, 0.6)
        self.assertEqual(config.TTS_TOP_K, 40)
        self.assertEqual(config.TTS_TOP_P, 0.8)

    def test_default_animation_prompt_describes_idle_motion(self):
        prompt = generate._resolve_animation(None)
        lowered = prompt.lower()

        self.assertIn("pauses", lowered)
        self.assertIn("silence", lowered)
        self.assertIn("upright", lowered)
        self.assertIn("breathing", lowered)
        self.assertIn("blinking", lowered)
        self.assertIn("drooping", lowered)
        self.assertGreaterEqual(len(prompt.split()), 50)

    def test_video_generation_uses_animation_prompt_for_infinitetalk(self):
        source = Path("video_engine.py").read_text(encoding="utf-8")

        self.assertIn("prompt = animation_prompt if animation_prompt else self._PROMPT", source)
        self.assertIn('"prompt": prompt', source)
        self.assertIn("self._pre_encode_text(prompt)", source)
        self.assertIn("pre_encoded_text=self._cached_text_embeddings", source)


if __name__ == "__main__":
    unittest.main()
