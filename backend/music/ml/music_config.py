

PITCH_CLASS_VOCAB = 12
OCTAVE_VOCAB = 11
PITCH_VOCAB = 128
VEL_VOCAB = 9
DT_VOCAB = 64
DUR_VOCAB = 2
SUS_VOCAB = 2   # sustain pedal (CC64): off / on

MAX_PITCH = PITCH_VOCAB - 1
MAX_VELOCITY = VEL_VOCAB - 1
MAX_TIME = DT_VOCAB - 1
MAX_DURATION = DUR_VOCAB - 1
MAX_SUSTAIN = SUS_VOCAB - 1

# DT_MAX_SECONDS = 2.0 # todo: this should be set at fine tuning. fine tune song gets parsed and its dt_max should be used on the generated song

TARGET_SECONDS = 60  # fixed generation length.
