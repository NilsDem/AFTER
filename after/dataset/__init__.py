from .dataset import SimpleDataset, CombinedDataset
from .lazy_waveform import (CombinedLazyWaveformDataset, LazyWaveformDataset,
                            is_lazy_waveform_dataset)
from .transforms import random_phase_mangle
from .audio_example import AudioExample
#from .utils import get_beat_signal
#from .beat_this import inference
#from .beat_this.inference import Audio2Beats, File2Beats
