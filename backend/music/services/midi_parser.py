
import math
import mido
from ..ml import music_config
from . import midi_tester
import os
from pathlib import Path


# This is only intended for piano/ single instrument music!

MAX_MIDI_PITCH = 127
MAX_MIDI_VELOCITY = 127

# INFO: Tempo is put into times. Every MIDI event has time. the time difference between events is time * current relativeTempo

# off-notes aren't modeled/generated - convertedNotes() derives a duration heuristically
# instead: pedal down holds almost the whole gap to the next note (legato), pedal up
# releases a bit earlier (detached)
SUSTAIN_DURATION_FRACTION = 0.95
STACCATO_DURATION_FRACTION = 0.7
MIN_NOTE_DURATION_SECONDS = 0.05


class MidiParser:
    def __init__(self, MAX_VELOCITY = music_config.MAX_VELOCITY, MAX_TIME = music_config.MAX_TIME, MAX_DURATION=music_config.MAX_DURATION, MAX_PITCH=music_config.MAX_PITCH, MAX_SUSTAIN=music_config.MAX_SUSTAIN ):
        self.MAX_VELOCITY = MAX_VELOCITY
        self.MAX_TIME = MAX_TIME
        self.MAX_DURATION = MAX_DURATION
        self.MAX_PITCH = MAX_PITCH
        self.MAX_SUSTAIN = MAX_SUSTAIN


    # convert MIDI into feature vectors: pitch, velocity, time. Time = time since previous note started
    def read_midi(self, midi_file_path):
        if isinstance(midi_file_path, (str, os.PathLike)):
            mid = mido.MidiFile(midi_file_path)
        else:
            mid = mido.MidiFile(file=midi_file_path)

        current_time = 0
        startTime = False # Start measuring time at first note event
        self.midi_data = [] # put feature vectors here [pitch, velocity, time]
        self.meta_data = [] # meta data. todo: need this?

        maxTime = 0.0 # time needs to be quantized.

        startTempo = 500000 # default tempo
        currentTempo = startTempo # todo: need start and current?
        previousTime = 0
        relativeTempo = 1

        volume = 100 # default values
        sustain = 0

        for msg in mid:
            noteType = msg.type

            if( not startTime and noteType == "note_on" ):
                startTime = True

            if startTime:
                current_time += msg.time

            if noteType == 'set_tempo':
                currentTempo = msg.tempo

            if noteType != 'note_on' and noteType != 'note_off':
                if( noteType != 'control_change' ):
                    continue

                if( msg.control == 7 ):
                    volume = msg.value

                if( msg.control == 64 ):
                    sustain = msg.value

                continue

            if msg.velocity == 0 and noteType == 'note_on':
                noteType = 'note_off' # some MIDI files have onNotes with velocity 0 instead of offNotes

            if noteType != "note_on":
                continue # note_off carries no data we need since we don't track duration

            deltaTime = ( current_time - previousTime ) * relativeTempo
            previousTime = current_time

            # fold channel volume (CC7) into note velocity as a single "effective loudness" value
            combinedVelocity = min(MAX_MIDI_VELOCITY, ( msg.velocity * volume ) // MAX_MIDI_VELOCITY )
            velocity = ( combinedVelocity // ( MAX_MIDI_VELOCITY // self.MAX_VELOCITY ) )  # max is 8

            sustainBin = 1 if sustain >= 64 else 0  # CC64 convention: >=64 is pedal-down

            if( maxTime < deltaTime ):
                maxTime = deltaTime

            self.midi_data.append([msg.note, velocity, deltaTime, sustainBin, current_time])  # ( note, velocity, delta time, sustain, absolute time )

            relativeTempo = startTempo / currentTempo

        maxTime = maxTime * 0.95 # cut off too long times whet putting time into bins

        self.songTime = current_time

        timeSum = 0
        count = 0
        for data in self.midi_data:
            # print("data:", data)
            t = min(data[2], maxTime)

            timeSum += t
            count += 1

            data[2] = int(self.MAX_TIME * t // maxTime)
            data[4] = ( data[4] / self.songTime ) if self.songTime > 0 else 0.0  # 0 (start) .. 1 (end)

        return timeSum / count # used to set the speed to original speed


        # test:
        # if isinstance(midi_file_path, (str, os.PathLike)):
        #     test_name = os.path.basename(midi_file_path)
        # else:
        #     test_name = "uploaded.mid"
        # output_path = os.path.join("tests", test_name)
        #
        # midi_tester.testMidi(
        #     self.convertedNotes(self.midi_data),
        #     output_path
        # )


    def convertedNotes(self, generatedNotes, averageTime = 0): # this is used after transformer
        converted = []
        timeSum = 0
        count = 0

        for _, _, dt, _ in generatedNotes:
            timeSum += dt
            count += 1


        if( timeSum <= 0 ):
            raise ValueError( f"Time sum of generated music is 0" )

        timeFactor = averageTime / (timeSum/count) if (timeSum > 0 and count > 0) else 1 # timeFactor is used to make speed of song closer to original

        # real gap (seconds) since the previous note, for every note
        times = [timeFactor * dt for _, _, dt, _ in generatedNotes]
        fallbackDuration = averageTime if averageTime > 0 else 0.5

        for i, (p, v, dt, sustain) in enumerate(generatedNotes):
            velocity = v * (MAX_MIDI_VELOCITY // self.MAX_VELOCITY)
            time = times[i]
            sustainValue = sustain * MAX_MIDI_VELOCITY  # binary -> 0 or 127

            nextGap = times[i + 1] if i + 1 < len(times) else fallbackDuration
            fraction = SUSTAIN_DURATION_FRACTION if sustain else STACCATO_DURATION_FRACTION
            duration = max(MIN_NOTE_DURATION_SECONDS, nextGap * fraction)

            converted.append((p, velocity, time, sustainValue, duration))

        return converted

def readMidiFiles(midiDir):

    songs = []
    parser = MidiParser()

    for file in Path(midiDir).iterdir():

        if file.is_file():
            filepath = os.path.join(midiDir, file.name)
            parser.read_midi(filepath)

            # midi_tester.testMidi( parser.convertedNotes( parser.midi_data ) )
            songs.append(parser.midi_data)

    return songs

# todo: generated notes dont have offnotes? - see https://spessasus.github.io/SpessaSynth/
# todo: make notes turn off after some time!