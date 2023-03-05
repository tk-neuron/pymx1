import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class NeuronManager:
    # mapping: pd.DataFrame  # neuronal unit
    group: int
    peaks: np.ndarray  # peaks
    # waveforms: Optional[np.ndarray]  # waveforms
    template: Optional[np.ndarray]
    # x: Optional[float]
    # y: Optional[float]

    # def get_template(self):
    #     template = np.mean(self.waveforms, axis=0)
    #     self.template = template

    # def get_location(self):
    #     if self.template is None:
    #         self.get_template()

    #     amps = np.max(self.template, axis=1) - np.min(self.template, axis=1)
    #     electrode_ = self.mapping.iloc[np.argmax(amps)]
    #     self.x, self.y = electrode_[['x', 'y']].values
