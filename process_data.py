import pandas as pd
import os
import shutil

##########BEETHOVEN##########
# beethoven_df = pd.read_csv("./data/maestro_beethoven.csv", index_col=False)

# midi_files_df = beethoven_df["midi_filename"]
# midi_files_list = midi_files_df.to_list()

# beethoven_files = []
# rootdir = './maestro-v3.0.0'

# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         if file in midi_files_list:
#             shutil.copy(os.path.join(subdir, file), os.path.join("/Users/anessapetteruti/Desktop/Bot-hoven/data/maestro_beethoven", file))
           

##########BACH##########
# bach_df = pd.read_csv("./data/maestro_bach.csv", index_col=False)

# midi_files_df = bach_df["midi_filename"]
# midi_files_list = midi_files_df.to_list()

# bach_files = []
# rootdir = './maestro-v3.0.0'

# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         if file in midi_files_list:
#             shutil.copy(os.path.join(subdir, file), os.path.join("/Users/anessapetteruti/Desktop/Bot-hoven/data/maestro_bach", file))

##########SCHUBERT##########
# schubert_df = pd.read_csv("./data/maestro_schubert.csv", index_col=False)

# midi_files_df = schubert_df["midi_filename"]
# midi_files_list = midi_files_df.to_list()

# schubert_files = []
# rootdir = './maestro-v3.0.0'

# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         if file in midi_files_list:
#             shutil.copy(os.path.join(subdir, file), os.path.join("/Users/anessapetteruti/Desktop/Bot-hoven/data/maestro_schubert", file))

##########ROMANTIC##########
# romantic_df = pd.read_csv("./data/maestro_romantic.csv", index_col=False)

# midi_files_df = romantic_df["midi_filename"]
# midi_files_list = midi_files_df.to_list()

# romantic_files = []
# rootdir = './maestro-v3.0.0'

# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         if file in midi_files_list:
#             shutil.copy(os.path.join(subdir, file), os.path.join("/Users/anessapetteruti/Desktop/Bot-hoven/data/maestro_romantic", file))

##########SMALL ROMANTIC##########
# small_romantic_df = pd.read_csv("./data/small_romantic.csv", index_col=False)

# midi_files_df = small_romantic_df["midi_filename"]
# midi_files_list = midi_files_df.to_list()

# small_romantic_files = []
# rootdir = './maestro-v3.0.0'

# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         if file in midi_files_list:
#             shutil.copy(os.path.join(subdir, file), os.path.join("/Users/anessapetteruti/Desktop/Bot-hoven/data/small_romantic", file))


##########MEDIUM ROMANTIC##########
medium_romantic_df = pd.read_csv("./data/medium_romantic.csv", index_col=False)

midi_files_df = medium_romantic_df["midi_filename"]
midi_files_list = midi_files_df.to_list()

medium_romantic_files = []
rootdir = './maestro-v3.0.0'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file in midi_files_list:
            shutil.copy(os.path.join(subdir, file), os.path.join("/Users/anessapetteruti/Desktop/Bot-hoven/data/medium_romantic", file))