from enum import Enum

from . import midi_parser as Parser
from . import midi_tester

from ..ml import rnn
from ..ml import music_transformerT1 as tr1
from ..ml import music_transformerT2 as tr2

import io

class ModelNames(Enum):
    RNN = "rnn"
    TRANSFORMER1 = "transformer1"
    TRANSFORMER2 = "transformer2"

    @classmethod
    def allModels(cls):
        return [model.value for model in cls]

class Composer:
    def __init__(self):
        self.currentModelName = "Not Selected"
        # self.currentModelName = ModelNames.TRANSFORMER2
        # self.currentModel = self.setModel(self.currentModelName)

    def selectModel(self, model: ModelNames):
        self.currentModelName = model
        self.currentModel = self.setModel(self.currentModelName)
        self.currentModel.fineTune(self.currentSong) # todo: also store tempo of the song in composer. the generated song should have the same tempo
        print( "end of selectModel. Fine tuning complete" )

    def setModel(self, model: ModelNames):
        match model:
            case ModelNames.RNN:
                print("Loading RNN")
                return rnn.loadModel()

            case ModelNames.TRANSFORMER1:
                print("Loading Transformer 1")
                return tr1.loadModel()

            case ModelNames.TRANSFORMER2:
                print("Loading Transformer 2")
                return tr2.loadModel()

            case _:
                raise ValueError("Unknown model")

    def getModelName(self):
        return self.currentModelName

    def handleUpload( self, content ):

        print("Size of midi file:", len(content))

        midi_stream = io.BytesIO(content)

        print("Parsing MIDI from memory...")

        parser = Parser.MidiParser()
        self.maxTime = parser.read_midi(midi_stream)

        self.currentSong = parser.midi_data

    def generateMusic(self):
        print("Generating Music begin")
        generatedNotes = self.currentModel.generate(self.currentSong) #, maxTime=self.maxTime)
        parser = Parser.MidiParser()
        convertedNotes = parser.convertNotes(generatedNotes, self.maxTime)
        print("Generating Music end")
        return midi_tester.midiToBytes(convertedNotes)


composer = Composer()