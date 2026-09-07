from abc import ABC, abstractmethod
import torch.nn as nn

# number of trailing notes from the uploaded seed song used to prompt generation
SEED_NOTES = 50


class BaseMusicModel(nn.Module, ABC):
  @abstractmethod
  def fineTune(self, song): ...


  @abstractmethod
  def generate(self, seedSong): ...

