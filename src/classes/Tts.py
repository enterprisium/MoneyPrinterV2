import os
from config import ROOT_DIR
from TTS.api import TTS as XTTS_V2

class TTS:
    """
    Class for Text-to-Speech using XTTS_V2.
    """
    def __init__(self, language: str = "en", voice: str = "default") -> None:
        """
        Initializes the TTS class.

        Args:
            language (str): The language for TTS. Defaults to "en".
            voice (str): The voice for TTS. Defaults to "default".

        Returns:
            None
        """
        self.language = language
        self.voice = voice
        self.synthesizer = XTTS_V2(language=self.language, voice=self.voice)

    def synthesize(self, text: str, output_file: str = os.path.join(ROOT_DIR, ".mp", "audio.wav")) -> str:
        """
        Synthesizes the given text into speech.

        Args:
            text (str): The text to synthesize.
            output_file (str, optional): The output file to save the synthesized speech. Defaults to "audio.wav".

        Returns:
            str: The path to the output file.
        """
        # Synthesize the text
        outputs = self.synthesizer.tts(text)

        # Save the synthesized speech to the output file
        self.synthesizer.save_wav(outputs, output_file)

        return output_file
