import os
import glob
import pretty_midi

from tqdm import tqdm

from src.config import (
    RAW_MIDI_DIR,
    genres
)


class MIDIParser:

    def __init__(self, root_dir=RAW_MIDI_DIR):

        self.root_dir = root_dir

    def get_all_midi_files(self):

        midi_files = []

        for ext in ["*.mid", "*.midi"]:

            midi_files.extend(
                glob.glob(
                    str(self.root_dir / "**" / ext),
                    recursive=True
                )
            )

        return sorted(midi_files)

    def get_genre_from_path(self, midi_path):

        path_parts = midi_path.split(os.sep)

        for genre in genres:

            if genre in path_parts:

                return genre

        return "unknown"

    def parse_midi(self, midi_path):

        try:

            midi_data = pretty_midi.PrettyMIDI(midi_path)

            notes = []

            for instrument in midi_data.instruments:

                if instrument.is_drum:
                    continue

                for note in instrument.notes:

                    notes.append({
                        "pitch": note.pitch,
                        "velocity": note.velocity,
                        "start": note.start,
                        "end": note.end,
                        "duration": note.end - note.start
                    })

            info = {
                "path": midi_path,
                "genre": self.get_genre_from_path(midi_path),
                "n_instruments": len(midi_data.instruments),
                "tempo": midi_data.estimate_tempo(),
                "duration": midi_data.get_end_time(),
                "time_signature_changes": len(
                    midi_data.time_signature_changes
                ),
                "key_signature_changes": len(
                    midi_data.key_signature_changes
                ),
                "notes_count": len(notes),
                "notes": notes
            }

            return info

        except Exception as e:

            print(f"[ERROR] Failed to parse: {midi_path}")

            print(e)

            return None

    def build_metadata(self):

        midi_files = self.get_all_midi_files()

        metadata = []

        print(f"\nFound {len(midi_files)} MIDI files\n")

        for midi_path in tqdm(
            midi_files,
            desc="Parsing MIDI Files"
        ):

            data = self.parse_midi(midi_path)

            if data is not None:

                metadata.append(data)

        return metadata


def load_midi(midi_path):

    return pretty_midi.PrettyMIDI(midi_path)


def parse_dataset(root_dir=RAW_MIDI_DIR):

    parser = MIDIParser(root_dir=root_dir)

    return parser.build_metadata()


def get_midi_files(root_dir=RAW_MIDI_DIR):

    parser = MIDIParser(root_dir=root_dir)

    return parser.get_all_midi_files()


if __name__ == "__main__":

    parser = MIDIParser()

    metadata = parser.build_metadata()

    print("\nParsed MIDI Files:\n")

    for item in metadata[:5]:

        print(item)
