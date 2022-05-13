import numpy as np
import argparse
import pathlib


if __name__ == '__main__':

    ext_list = ['*.midi', '*.mid']

    midi_filenames = []
    for ext in ext_list:
        ext_filenames = pathlib.Path(args.midi_dir).rglob(ext)
        ext_filenames = list(map(lambda x: str(x), ext_filenames))
        midi_filenames += ext_filenames
    print(f'Found {len(midi_filenames)} midi files')
    assert len(midi_filenames) > 0

    if not args.n_files is None:
        n_files = max(0, min(args.n_files, len(midi_filenames)))
        midi_filenames = np.random.choice(
            midi_filenames, n_files, replace=False)
        assert len(midi_filenames) > 0

    idx_to_time = get_quant_time()

    midi_parser = MIDI_parser.build_from_config(config, idx_to_time)

    print('Creating npz files...')
    midi_parser.preprocess_dataset(src_filenames=midi_filenames,
                                   dst_dir=args.npz_dir, batch_size=20, dst_filenames=None)

    print(f'Created dataset with {len(midi_filenames)} files')
