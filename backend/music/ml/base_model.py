from abc import ABC, abstractmethod
import torch.nn as nn

# number of notes from the start of the uploaded seed song kept as a real anchor to
# kick off generation - everything after it is composed by the model, so the output is
# a full song (matching the seed song's note count) rather than a continuation of it
SEED_NOTES = 1


class BaseMusicModel(nn.Module, ABC):
  @abstractmethod
  def fineTune(self, song): ...


  @abstractmethod
  def generate(self, seedSong): ...

