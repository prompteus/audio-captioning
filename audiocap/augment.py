import warnings
from typing import NamedTuple

import audiomentations


class AugmentConfig(NamedTuple):
    p_total: float = 0.0
    p_noise: float = 0.3
    p_shift: float = 0.3
    p_gain: float = 0.3


class Augmenter:

    def __init__(
        self,
        config: AugmentConfig,
    ) -> None:
        self.config = config
        augmentations = [
            audiomentations.AddGaussianSNR(min_snr_in_db=40.0, max_snr_in_db=100.0, p=config.p_noise),
            audiomentations.Shift(min_fraction=-0.1, max_fraction=0.1, rollover=True, p=config.p_shift),
            audiomentations.Compose([
                audiomentations.Gain(min_gain_in_db=10, max_gain_in_db=10, p=1),
                audiomentations.Clip(a_min=-1.0, a_max=1.0, p=1)
            ], p=config.p_gain),
        ]

        self.augment = audiomentations.Compose(augmentations, p=config.p_total)

    def __call__(self, data, sample_rate):
        try: 
            data = self.augment(data, sample_rate)
        except:
            warnings.warn("Augmentation failed, returning original data")
        return data
