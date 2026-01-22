import miditoolkit
import sys

nomFichier="../GrandMidiPiano/Bach, Johann Sebastian, Partita in G major, BWV 829, aecenXn3obw.mid"
midi = miditoolkit.MidiFile(nomFichier)
try:
    sortie = sys.argv[1]
except:
    sortie = "test.txt"


tokens = []

for inst in midi.instruments:
    for note in inst.notes:
        tokens.append(f"POSITION_{note.start}")
        tokens.append(f"NOTE_ON_{note.pitch}")
        tokens.append(f"VELOCITY_{note.pitch}")
        tokens.append(f"DURATION_{note.end - note.start}")

with open(nomFichier, "w", encoding="utf-8") as f:
    for token in tokens:
        f.write(token+"\n")
f.close()