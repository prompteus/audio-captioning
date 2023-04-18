from __future__ import annotations
import collections
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class AudiosetOntology:
    def __init__(self, df: pd.DataFrame):
        assert set(df.columns) == {'name', 'description', 'citation_uri', 'positive_examples', 'child_ids', 'restrictions'}
        self.df = df

        children = df["child_ids"].explode().dropna().rename("child_id")
        parents = pd.DataFrame({"child_id": children.values, "parent_id": children.index})
        
        self.parents: dict[str, list[str]]
        self.parents = parents.groupby("child_id")["parent_id"].apply(list).rename("parent_ids").to_dict()
        for label_id in self.df.index:
            self.parents[label_id] = self.parents.get(label_id, [])

    @staticmethod
    def from_json_file(file_path) -> "AudiosetOntology":
        ontology = pd.read_json(file_path).set_index('id', drop=True)
        return AudiosetOntology(ontology)    
    
    def audioset_label_ids_to_str(
        self,
        label_ids: str | list[str],
        include_parents: bool = True,
    ) -> str:
        """
        Converts a list of Audioset label ids to a string.
        Does not support batched inputs.
        """
        
        if isinstance(label_ids, str):
            label_ids = label_ids.split(",")

        label_ids = [label.strip().strip('"') for label in label_ids]
        names = []

        for label_id in label_ids:
            if include_parents:
                parent_ids = self.parents[label_id]
                for parent_id in parent_ids:
                    name = self.df.loc[parent_id]["name"]
                    if name not in names:
                        names.append(name)

            name = self.df.loc[label_id]["name"]
            if name not in names:
                names.append(name)

        return ", ".join([name.replace(",", " -").replace(":", " -").lower() for name in names])
    
    def all_descendants(self, *, label_id: str | None = None, name: str | None = None, include_self: bool) -> pd.DataFrame:
        """
        Finds recursively all descendants of a label according to the ontology.
        """
        if label_id is None:
            assert name is not None
            label_id = self.name_to_id(name)
        descendants = set()
        if include_self:
            descendants.add(label_id)
        for child_id in self.df.loc[label_id]["child_ids"]:
            descendants.update(self.all_descendants(label_id=child_id, include_self=True).index)
        return self.df.loc[list(descendants)]
    
    def name_to_id(self, name: str) -> str:
        return str(self.df[self.df["name"] == name].index[0])


class AudiosetSubsetSelector:
    def __init__(self, *, df: pd.DataFrame, desired_num_samples_per_class: dict[str, int], seed: int, step: int, ontology: AudiosetOntology):
        self.ontology = ontology
        self.labels_exploded = df.explode("labels")
        self.inverse_index = self.labels_exploded.groupby("labels")["youtube_id"].agg(list).to_dict()
        self.df = df.set_index("youtube_id")
        self.df["selected"] = False

        music_id = ontology.name_to_id("Music")
        self.df["contains_music"] = self.df["labels"].apply(lambda labels: music_id in labels)

        speech_id = ontology.name_to_id("Speech")
        self.df["contains_speech"] = self.df["labels"].apply(lambda labels: speech_id in labels)

        self.df["contains_no_music_or_speech"] = ~(self.df["contains_music"] | self.df["contains_speech"])
        self.df["contains_music_xor_speech"] = self.df["contains_music"] ^ self.df["contains_speech"]
        self.df["contains_music_and_speech"] = self.df["contains_music"] & self.df["contains_speech"]

        self.desired_num_samples_per_class = desired_num_samples_per_class
        self.seed = seed
        self.step = step
        self.all_labels = set(self.labels_exploded["labels"].unique())
        self.total_label_counts = self.labels_exploded["labels"].value_counts().to_dict()
        self.label_counts = collections.Counter({label: 0 for label in self.all_labels}) 
        self.random_generator = random.Random(self.seed)
        # it's important to make label_counts Counter,
        # because update then adds the values, not overwrites them

    def select_reasonable_subset(self) -> pd.DataFrame:
        queue = collections.deque(sorted(self.all_labels, key=lambda label: self.total_label_counts[label]))
        max_steps = sum(min(self.desired_num_samples_per_class[label], self.total_label_counts[label]) for label in self.all_labels)

        with tqdm(total=max_steps) as pbar:
            while queue:
                label = queue.popleft()

                if self.label_counts[label] > self.total_label_counts[label]:
                    raise ValueError("Should not happen")
                
                if self.label_counts[label] == self.total_label_counts[label]:
                    continue
                
                if self.label_counts[label] >= self.desired_num_samples_per_class[label]:
                    continue

                if self.total_label_counts[label] <= self.desired_num_samples_per_class[label]:
                    num_selected = self.select_random_sample(label, limit=None)
                    pbar.update(num_selected)
                    continue

                missing_size = self.desired_num_samples_per_class[label] - self.label_counts[label]
                limit = min(missing_size, self.step)
                num_selected = self.select_random_sample(label, limit=limit)
                pbar.update(num_selected)
                if num_selected != 0:
                    queue.append(label)

        return self.df[self.df["selected"]]


    def select_random_sample(self, label: str, limit: int | None) -> int:
        # select not selected clips that contain the label using inverse index
        containing_clips: pd.DataFrame = self.df.loc[self.inverse_index[label]]
        containing_clips = containing_clips[~containing_clips["selected"]]

        if limit is not None:
            containing_clips = self.limit_speech_and_music(containing_clips, limit)
            if len(containing_clips) > limit:
                seed = self.random_generator.randint(0, 100000)
                containing_clips = containing_clips.sample(limit, random_state=seed)
        self.df.loc[containing_clips.index, "selected"] = True
        self.label_counts.update(containing_clips["labels"].explode().value_counts().to_dict())
        return len(containing_clips)

    def limit_speech_and_music(self, clips: pd.DataFrame, limit: int) -> pd.DataFrame:
        contains_none = clips[clips["contains_no_music_or_speech"]]
        if len(contains_none) >= limit:
            return contains_none

        seed = self.random_generator.randint(0, 10000)
        contains_one = clips[clips["contains_music_xor_speech"]]
        if len(contains_one) + len(contains_none) >= limit:
            missing = limit - len(contains_none)
            sample = contains_one.sample(min(missing, len(contains_one)), random_state=seed)
            return pd.concat([contains_none, sample])
        
        contains_both = clips[clips["contains_music_and_speech"]]
        missing = limit - len(contains_none) - len(contains_one)
        sample = contains_both.sample(min(missing, len(contains_both)), random_state=seed)
        return pd.concat([contains_none, contains_one, sample])


def balanced_split(
    df: pd.DataFrame,
    minimum_test_examples_per_class: int,
    must_be_in_train: set[str],
    must_be_in_test: set[str],
    seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.reset_index()
    assert "labels" in df.columns
    assert "youtube_id" in df.columns
    inverse_index = df.explode("labels").groupby("labels").agg({"youtube_id": list})
    df = df.set_index("youtube_id")

    test = must_be_in_test
    test_freqs = collections.Counter({label: 0 for label in inverse_index.index})
    for youtube_id in test:
        test_freqs.update(df.loc[youtube_id, "labels"])

    random_generator = np.random.default_rng(seed)
    for label, youtube_ids in inverse_index.itertuples(index=True):
        if len(youtube_ids) < minimum_test_examples_per_class:
            continue
        youtube_ids = list(set(youtube_ids) - must_be_in_train)
        test.update(random_generator.choice(youtube_ids, minimum_test_examples_per_class, replace=False))
        test_freqs.update({label: minimum_test_examples_per_class})
    df = df.reset_index()
    train = set(df["youtube_id"]) - test
    return df[df["youtube_id"].isin(train)], df[df["youtube_id"].isin(test)]

