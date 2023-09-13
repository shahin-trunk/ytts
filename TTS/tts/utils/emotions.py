from typing import List

from coqpit import Coqpit

from TTS.config import check_config_and_model_args
from TTS.tts.utils.managers import BaseIDManager


class EmotionManager(BaseIDManager):
    def __init__(self, emotion_ids_file_path: str = ""):
        super().__init__(id_file_path=emotion_ids_file_path)

    @property
    def num_emotions(self) -> int:
        return len(list(self.name_to_id.keys()))

    @property
    def emotion_names(self) -> List:
        return list(self.name_to_id.keys())

    def save_ids_to_file(self, file_path: str) -> None:
        """Save language IDs to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        self._save_json(file_path, self.name_to_id)

    @staticmethod
    def init_from_config(config: Coqpit) -> "EmotionManager":
        """Initialize the language manager from a Coqpit config.

        Args:
            config (Coqpit): Coqpit config.
        """
        emotion_manager = None
        if check_config_and_model_args(config, "use_emotion_embedding", True):
            if config.get("language_ids_file", None):
                emotion_manager = EmotionManager(emotion_ids_file_path=config.emotion_ids_file)

        return emotion_manager
