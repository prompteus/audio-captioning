import warnings
from typing import NamedTuple

import audiomentations


class AugmentConfig(NamedTuple):
    p_total: float = 0.0
    p_noise: float = 0.3
    p_pitch: float = 0.3
    p_stretch: float = 0.3
    p_shift: float = 0.3


class Augmenter:

    def __init__(
        self,
        config: AugmentConfig,
    ) -> None:
        self.config = config
        augmentations = [
            audiomentations.AddGaussianSNR(min_snr_in_db=30.0, max_snr_in_db=100.0, p=config.p_noise),
            audiomentations.PitchShift(min_semitones=-4, max_semitones=4, p=config.p_pitch),
            audiomentations.TimeStretch(min_rate=0.9, max_rate=1.1, leave_length_unchanged=True, p=config.p_stretch),
            audiomentations.Shift(min_fraction=-0.25, max_fraction=0.25, rollover=True, p=config.p_shift),
        ]
        self.augment = audiomentations.Compose(augmentations, p=config.p_total)


    def __call__(self, data, sample_rate):
        try: 
            data = self.augment(data, sample_rate)
        except:
            warnings.warn("Augmentation failed, returning original data")

        return data
