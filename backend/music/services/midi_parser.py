
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


class MidiParser:
    def __init__(self, MAX_VELOCITY = music_config.MAX_VELOCITY, MAX_TIME = music_config.MAX_TIME, MAX_DURATION=music_config.MAX_DURATION, MAX_PITCH=music_config.MAX_PITCH ):
        self.MAX_VELOCITY = MAX_VELOCITY
        self.MAX_TIME = MAX_TIME
        self.MAX_DURATION = MAX_DURATION
        self.MAX_PITCH = MAX_PITCH


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

        startTempo = -1
        currentTempo = 1
        previousTime = 0
        relativeTempo = 1

        for msg in mid:
            # print( "        Midi msg: " + str(msg) )
            # todo: handle different channels? - have option to set only main channel

            noteType = msg.type

            if( not startTime and noteType == "note_on" ):
                startTime = True

            if startTime:
                current_time += msg.time

            if noteType == 'set_tempo':
                # print("tempo change!!!!: ", msg.tempo)
                if startTempo == -1:
                    startTempo = msg.tempo
                currentTempo = msg.tempo

            if noteType != 'note_on' and noteType != 'note_off':

                self.meta_data.append( msg )
                # todo: include control change 64. (sustain). also check if note numbering is ok. include sustain in midi_data
                # todo: also include cc 7 (volume)
                # todo: set tempo:  Midi msg: MetaMessage('set_tempo', tempo=983606, time=0.03225803333333333)
                # todo: tempo change can be put in times. read start tempo in beginning. then change times: time = time * startTempo/currentTempo
                continue

            if msg.velocity == 0 and noteType == 'note_on':
                noteType = 'note_off' # some MIDI files have onNotes with velocity 0 instead of offNotes

            if noteType != "note_on":
                continue # note_off carries no data we need since we don't track duration

            deltaTime = ( current_time - previousTime ) * relativeTempo
            previousTime = current_time

            velocity = ( msg.velocity // ( MAX_MIDI_VELOCITY // self.MAX_VELOCITY ) )  # max is 8

            if( maxTime < deltaTime ):
                maxTime = deltaTime

            self.midi_data.append([msg.note, velocity, deltaTime])  # ( note, velocity, delta time )

            relativeTempo = startTempo / currentTempo

        maxTime = maxTime * 0.95 # cut off too long times whet putting time into bins

        for data in self.midi_data:
            # print("data:", data)
            t = min(data[2], maxTime)
            data[2] = int(self.MAX_TIME * t // maxTime)

        # todo: note times should be embeded relative to the average tempo!

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


    def convertedNotes(self, generatedNotes): # todo: this is used after transformer. make so everything is universal
        # print( "in convertedNotes" )
        converted = []
        for p, v, dt in generatedNotes:
            velocity = v * (MAX_MIDI_VELOCITY // self.MAX_VELOCITY)
            time = dt * music_config.DT_MAX_SECONDS / self.MAX_TIME
            converted.append((p, velocity, time))

        # print("end of ConvertedNotes")
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

        # break # test delete

    return songs

# todo: generated notes dont have offnotes? - see https://spessasus.github.io/SpessaSynth/
# make notes turn off after some time!