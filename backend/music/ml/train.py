from pathlib import Path

from ..services.midi_parser import readMidiFiles

from . import rnn
from . import music_transformerT1 as tr1
from . import music_transformerT2 as tr2

MIDI_FILES_DIR = Path(__file__).resolve().parent.parent / "midiFiles"
SONGS_DIRS = [
    MIDI_FILES_DIR / "midiFavourites",
    MIDI_FILES_DIR / "piano-midi",
    MIDI_FILES_DIR / "maestro" / "maestro-v3.0.0",
]


def train():
    songs = []
    for songsDir in SONGS_DIRS:
        print(f"Reading {songsDir}", flush=True)
        songs += readMidiFiles(songsDir)

    print(f"Read {len(songs)} songs total", flush=True)

    # print("Training RNN...", flush=True)
    # rnn.trainModel(songs)
    #
    # print("Training TransformerT1...", flush=True)
    # tr1.trainModel(songs)

    print("Training TransformerT2...", flush=True)
    tr2.trainModel(songs)


if __name__ == "__main__":
    train()
