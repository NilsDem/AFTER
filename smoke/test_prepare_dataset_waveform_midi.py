"""MIDI extraction must work when preprocessing waveform-only datasets."""
import numpy as np
import pretty_midi

from after_scripts.prepare_dataset import load_or_extract_midi


def test_basic_pitch_is_independent_of_codec_embedding_model():
    expected = pretty_midi.PrettyMIDI()

    class FakeBasicPitch:
        def __init__(self):
            self.input_shape = None

        def __call__(self, audio):
            self.input_shape = audio.shape
            return expected

    extractor = FakeBasicPitch()
    stereo_audio = np.zeros((2, 1024), dtype=np.float32)
    result = load_or_extract_midi(None, extractor, stereo_audio, "audio.wav")
    assert result is expected
    assert extractor.input_shape == (1024, )


if __name__ == "__main__":
    test_basic_pitch_is_independent_of_codec_embedding_model()
