import sys
import miditoolkit
import re
# --- paramètres globaux ---
TPQ = 480  # ticks per quarter
TEMPO = 60

nomFichier=sys.argv[1]
file=open("../tests/"+nomFichier,"r")
contenuFichier=file.read()
contenuFichierSplite=contenuFichier.split("\n")

tokens=[]
for token in contenuFichierSplite:
    tokens.append(token)

# créer le MIDI
midi = miditoolkit.MidiFile(ticks_per_beat=TPQ)

# tempo
midi.tempo_changes.append(
    miditoolkit.TempoChange(TEMPO, time=0)
)


#tokens=["POSITION_2", "NOTE_ON_45", "DURATION_256", "POSITION_720", "NOTE_ON_48", "DURATION_244", "POSITION_960", "NOTE_ON_50", "DURATION_256", "POSITION_1200", "NOTE_ON_52", "DURATION_126", "POSITION_1680", "NOTE_ON_45", "DURATION_702", "POSITION_2400", "NOTE_ON_43", "DURATION_458", "POSITION_2880", "NOTE_ON_45", "DURATION_268", "POSITION_3840", "NOTE_ON_45", "DURATION_152", "POSITION_4560", "NOTE_ON_48", "DURATION_152", "POSITION_4800", "NOTE_ON_50", "DURATION_242", "POSITION_5040", "NOTE_ON_52", "DURATION_192", "POSITION_5520", "NOTE_ON_45", "DURATION_728", "POSITION_6240", "NOTE_ON_48", "DURATION_458", "POSITION_6720", "NOTE_ON_45", "DURATION_178", "POSITION_7680", "NOTE_ON_45", "DURATION_164", "POSITION_7920", "NOTE_ON_45", "DURATION_152", "POSITION_8400", "NOTE_ON_43", "DURATION_204", "POSITION_8640", "NOTE_ON_45", "DURATION_152", "POSITION_8880", "NOTE_ON_45", "DURATION_128", "POSITION_9360", "NOTE_ON_43", "DURATION_218", "POSITION_9600", "NOTE_ON_45", "DURATION_152", "POSITION_9840", "NOTE_ON_45", "DURATION_150", "POSITION_10320", "NOTE_ON_48", "DURATION_728", "POSITION_11040", "NOTE_ON_50", "DURATION_180", "POSITION_11520", "NOTE_ON_45", "DURATION_140", "POSITION_11760", "NOTE_ON_45", "DURATION_140", "POSITION_12240", "NOTE_ON_43", "DURATION_232", "POSITION_12480", "NOTE_ON_45", "DURATION_176", "POSITION_12720", "NOTE_ON_45", "DURATION_152", "POSITION_13200", "NOTE_ON_43", "DURATION_216", "POSITION_13440", "NOTE_ON_45", "DURATION_164", "POSITION_13680", "NOTE_ON_45", "DURATION_216", "POSITION_13920", "NOTE_ON_48", "DURATION_256"]


# instrument (piano)
instrument = miditoolkit.Instrument(
    program=0,
    is_drum=False,
    name="Piano"
)

current_time = 0
current_velocity = 80

i = 0
while i < len(tokens):
    token = tokens[i]

    if token == "BAR":
        # optionnel : on pourrait recalculer current_time
        i += 1
        continue

    elif token.startswith("POSITION_"):
        current_time = int(token.split("_")[1])
        i += 1


    elif token.startswith("NOTE_ON_"):
        pitch = int(token.split("_")[2])
        i += 1

    elif token.startswith("VELOCITY_"):
        current_velocity = int(token.split("_")[1])

        # on s'attend à un DURATION juste après
        next_token = tokens[i + 1]
        assert next_token.startswith("DURATION_")

        duration = int(next_token.split("_")[1])

        note = miditoolkit.Note(
            velocity=current_velocity,
            pitch=pitch,
            start=current_time,
            end=current_time + duration
        )

        instrument.notes.append(note)
        i += 2  # NOTE_ON + DURATION

    else:
        i += 1

# ajouter l'instrument
midi.instruments.append(instrument)

# sauvegarde
nomFichier = nomFichier.removesuffix(".txt")
midi.dump("../fichiers/tests_midi/"+nomFichier+".mid")
