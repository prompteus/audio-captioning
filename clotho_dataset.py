import base64
import glob
import os
import pickle
import re
from ast import literal_eval

import pandas as pd
from mutagen.wave import WAVE

global_params = {
    "dataset_dir": "data",
    "audio_splits": ["development", "validation", "evaluation", "test"],
    "text_files": ["development_captions.csv", "validation_captions.csv", "evaluation_captions.csv", "test_captions.csv"]
}

# %% 1. Process audio data

audio_fid2fname, audio_durations = {}, {}

for split in global_params["audio_splits"]:

    fid2fname, durations = {}, []

    audio_dir = os.path.join(global_params["dataset_dir"], split)

    for fpath in glob.glob(r"{}/*.wav".format(audio_dir)):
        try:
            clip = WAVE(fpath)

            if clip.info.length > 0.0:
                fid = base64.urlsafe_b64encode(os.urandom(8)).decode("ascii")
                fname = os.path.basename(fpath)

                fid2fname[fid] = fname
                durations.append(clip.info.length)
        except:
            print("Error audio file:", fpath)

    assert len(fid2fname) == len(durations)

    audio_fid2fname[split] = fid2fname
    audio_durations[split] = durations

# Save audio info
audio_info = os.path.join(global_params["dataset_dir"], "audio_info.pkl")
with open(audio_info, "wb") as store:
    pickle.dump({"audio_fid2fname": audio_fid2fname, "audio_durations": audio_durations}, store)
print("Save audio info to", audio_info)

# %% 2. Process text data

for split, text_fname in zip(global_params["audio_splits"], global_params["text_files"]):
    fid2fname = audio_fid2fname[split]
    stripped_fname2fid = {fid2fname[fid].strip(" "): fid for fid in fid2fname}

    text_fpath = os.path.join(global_params["dataset_dir"], text_fname)
    text_data = pd.read_csv(text_fpath, keep_default_na=False)

    text_rows = []

    for i in text_data.index:
        fname = text_data.iloc[i].fname
        raw_text = text_data.iloc[i].raw_text
        text = text_data.iloc[i].text

        fid = stripped_fname2fid[fname]
        tid = base64.urlsafe_b64encode(os.urandom(8)).decode("ascii")
        # Tokenize words with whitespaces
        tokens = [t for t in re.split(r"\s", text) if len(t) > 0]
        
        # tid, fid, fname, raw_text, text, tokens
        text_rows.append([tid, fid, fid2fname[fid], raw_text, text, tokens])

    text_rows = pd.DataFrame(data=text_rows, columns=["tid", "fid", "fname", "raw_text", "text", "tokens"])

    text_fpath = os.path.join(global_params["dataset_dir"], f"{split}_text.csv")
    text_rows.to_csv(text_fpath, index=False)
    print("Save", text_fpath)

# %% 3. Generate data statistics

# 1) audio data
# 2) text data
# 3) word frequencies
vocabulary = set()
word_bags = {}
split_infos = {}
for split, text_fname in zip(global_params["audio_splits"], global_params["text_files"]):
    fid2fname = audio_fid2fname[split]

    text_fpath = os.path.join(global_params["dataset_dir"], f"{split}_text.csv")
    text_data = pd.read_csv(text_fpath, converters={"tokens": literal_eval})

    num_clips = len(fid2fname)
    num_captions = text_data.tid.size

    bag = []
    for tokens in text_data["tokens"]:
        bag.extend(tokens)
        vocabulary = vocabulary.union(tokens)

    num_words = len(bag)
    word_bags[split] = bag
    split_infos[split] = {
        "num_clips": num_clips,
        "num_captions": num_captions,
        "num_words": num_words
    }

# Save vocabulary
vocab_info = os.path.join(global_params["dataset_dir"], "vocab_info.pkl")
with open(vocab_info, "wb") as store:
    pickle.dump({
        "vocabulary": vocabulary,
        "word_bags": word_bags,
        "split_infos": split_infos
    }, store)
print("Save vocabulary info to", vocab_info)