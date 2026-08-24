import io

import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from . import midi_parser as midiParser


# -----------------------------
# INPUT FORMAT
# (note, velocity, delta_time, duration)
# delta_time = time AFTER previous note ends
# duration = how long the note is held
# -----------------------------


# -----------------------------
# SETTINGS
# -----------------------------
TICKS_PER_BEAT = 480
BPM = 120

mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
track = MidiTrack()
mid.tracks.append(track)

# set tempo
tempo = mido.bpm2tempo(BPM)
track.append(MetaMessage('set_tempo', tempo=tempo, time=0))


def sec_to_ticks(seconds):
    beat_sec = 60 / BPM
    return int((seconds / beat_sec) * TICKS_PER_BEAT)


def _build_midi_file(notes):
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)

    # set tempo
    tempo = mido.bpm2tempo(BPM)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    # ----------------------------
    # build absolute event timeline
    # ----------------------------
    events = []
    current_time = 0
    last_sustain = None

    for note, velocity, delta_time, sustain in notes:

        current_time += sec_to_ticks(delta_time)
        start = current_time

        if sustain != last_sustain:
            events.append((start, 'control_change', 64, sustain))
            last_sustain = sustain

        events.append((start, 'note_on', note, velocity))
        events.append((1, 'note_off', note, 0)) # todo: 1 is duration. is this ok?

    # ----------------------------
    # sort events globally
    # ----------------------------
    events.sort(key=lambda x: x[0])

    # ----------------------------
    # convert to MIDI delta time stream
    # ----------------------------
    last_time = 0

    for time, msg_type, a, b in events:

        delta = time - last_time
        last_time = time

        if msg_type == 'control_change':
            track.append(Message('control_change', control=a, value=b, time=delta))
        else:
            track.append(Message(msg_type, note=a, velocity=b, time=delta))

    return mid


def testMidi(notes, fileName = "outputTest.mid"):
    print("midi testttttttttttttttttttttttt")

    mid = _build_midi_file(notes)

    # ----------------------------
    # save + playback
    # ----------------------------
    mid.save(fileName)
    print("saved:", fileName)


def midiToBytes(notes) -> bytes:
    mid = _build_midi_file(notes)

    buffer = io.BytesIO()
    mid.save(file=buffer)
    return buffer.getvalue()


def test():
    parser = midiParser.MidiParser()
    parser.read_midi("./midiFiles/maestro/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi")
    # midi_data stores velocity/delta_time as bin indices, not real MIDI values - convert back before playback
    testMidi(parser.convertedNotes(parser.midi_data), "test1.mid")

if __name__ == "__main__":
    test()

