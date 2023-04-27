# Audio captioning


## Setting up environment

Start by creating a conda environment:
```shell
git clone --recursive ... # recursive because there is `evaluation_tools` as git submodule
cd audio-captioning
conda create -n malach23 python=3.8
conda activate malach23
pip install -r requirements.txt
pip install -e .
```
If the last line does not work, update your pip. e.g. `pip install --upgrade pip`

After you have the environment ready, run the script inside audiocap/evaluation_tools
```
chmod +x audiocap/evaluation_tools/coco_caption/get_stanford_models.sh
./audiocap/evaluation_tools/coco_caption/get_stanford_models.sh
```
This will download the data necessary for computing evaluation metrics.


## Preparing data

We train on multiple datasets: Audioset (our selected subset), AudioCaps, and finally Clotho.
To make it simple to work with multiple datasets downloaded them to convert them into a file
structure that is as compatible as possible. We call it AudioFolder, because it is inspired
by HuggingFace's AudioFolder or ImageFolder.

While the datasets are not *completely* compatible (e.g. one caption vs multiple captions per
audio clip), AudioFolder structure and python class `audiocap.data.AudioFolder` helps us work
with them in a systematic way. The following sections explain how to get the data and prepare
AudioFolder from them.


### Clotho dataset

<details>
  <summary> Getting the data </summary>


```shell
mkdir -p data/clotho_v2.1/audiofolder
```

Download the data from <https://zenodo.org/record/4783391> and extract csv into the `data/clotho_v2.1` and audios into `data/clotho_v2.1/audiofolder` folder. Your tree structure should look like this:

```
audio-captioning/
├── audiocap
│   ...
...
|
├── data
│   └── clotho_v2.1
│       ├── audiofolder
│       │   ├─ development
│       │   ├─ evaluation
│       │   ├─ test
│       │   └─ validation
│       ├── clotho_captions_development.csv
│       ├── clotho_captions_evaluation.csv
│       ├── clotho_captions_validation.csv
│       ├── clotho_metadata_development.csv
│       ├── clotho_metadata_evaluation.csv
│       ├── clotho_metadata_test.csv
│       └── clotho_metadata_validation.csv
...
```

</details>


<details>
  <summary> Creating AudioFolder </summary>

Now, prepare 

```shell
python audiocap/prepare_audiofolder.py prepare-clotho-audiofolder data/clotho_v2.1/
```

This will prepare the folder into the format that is easily loadable.

To limit a size of a split (like validation and evaluation), run:

```shell
python audiocap/prepare_audiofolder.py limit-clotho-split data/clotho_v2.1/audiofolder/ validation --limit 200
python audiocap/prepare_audiofolder.py limit-clotho-split data/clotho_v2.1/audiofolder/ evaluation --limit 400
```

This will sample (with a seed) a subset with a desired size and move the remaining examples to the development split.

</details>


### Pretraining data

<details>
  <summary> Getting AudioSet </summary>

AudioSet is a large multi-label classification dataset. In our repository, we use information from
AudioSet ontology to construct keyword-based synthetic captions. This makes it possible to pretrain a
seq2seq captioning model (like Whisper) on AudioSet using an end-to-end supervised training pipeline.

AudioSet annotations are copied into this repository, but audios must be scraped from youtube.
You can use `scripts/download_audioset.sh` script that will use all cores to download and
convert audios based on youtube ids.

Make the script executable

```shell
chmod +x ./scripts/download_audioset.sh
```

Download the audio files

```shell
SPLIT='train_unbalanced' # run again with 'train_balanced' or 'eval'

mkdir -p logs/download_audioset

./scripts/download_audioset.sh \
    "data/audioset_full/csvs/${SPLIT}.csv" \
    "data/audioset_full/audios/${SPLIT}/" 2>&1 \
    | tee >( sed 's/.*\r//' > "logs/download_audioset/${SPLIT}.txt" )
```

(`sed` is there to delete output lines that just update the progress)

Please note that scraping AudioSet is best-effort only. Videos could be deleted from youtube.
Now, you should select a subset of AudioSet that suits your needs. AudioSet is heavily imbalanced,
with music and speech ocurring in a vast majority of examples. In our case, we selected
around 130k instances that covered as much of the underrepresented classes. However, before we
select the subset, we prepare AudioCaps - a different dataset we use for pretraining. This is
to prevent a leakage between the two datasets because they have audio files in common.

</details>


<details>
  <summary> Getting AudioCaps </summary>

AudioCaps is a captioning dataset with much more audios than Clotho (but is arguably of a lower quality).

AudioCaps annotations are also part of this repository. Furthermore, AudioCaps is a subset of AudioSet,
so you have all AudioCaps audios prepared once you download AudioSet.

</details>


<details>
  <summary> Creating AudioCaps AudioFolder </summary>

  Run:

  ```shell
    python audiocap/prepare_audiofolder.py prepare-audiocaps-audiofolder \
    --audiocaps-path data/audiocaps \
    --audioset-path data/audioset_full \
    --audio-format mp3
  ```

  This will copy the files from AudioSet, and prepare AudioFolder structure
  and annotations with dropped records about audios that were listed inside AudioCaps csvs
  but files were missing (unavailable when you scraped AudioSet).

</details>


<details>
  <summary> Creating a balanced AudioSet subset </summary>

  This part is most intricate. We want at the same time
  - a diverse subset
  - a balanced subset
  - a large subset
  - no leakeage with AudioCaps

  This is difficult and has no optimal solution. Especially balancing a dataset is difficult when each example has multiple labels.
  In this repository, there are some utilities help select it. If you want to select your own subset, you can look into `notebooks/select_audioset_subset.ipynb`

  However, the subset we selected is also available in this repository in `data/audioset_small`.

</details>


<details>
  <summary> Creating AudioSet-small AudioFolder </summary>

    Run:

  ```shell
    python audiocap/prepare_audiofolder.py prepare-audioset-small-audiofolder \
    --audioset-small-path data/audioset_small \
    --audioset-full-path data/audioset_full \
    --audio-format mp3
  ```

</details>

Congrats. Now you have all three datasets prepared for training.


### Checking corrupted audio files

During training, corrupted audio files (not loadable by librosa) are skipped.
However, if you want to check corrupted files, you can use the `audiocap.data.find_corrupted_audios`.


## Training

We train in two phases. We pretrain on a mixture of AudioCaps and AudioSet small, and
then finetune on Clotho.

We monitor metrics (into wandb) on each dataset separately and also log some predictions
so that one can see the outputs the model generates. 

Because we can pretrain using the same audio-to-text objective as we do on finetuning,
we can only have a single configurable training script.


### Pretraining 

AudioSet is originally a classification dataset. During training, we convert the labels on the fly
into keyword-based synthetic captions.

```shell
CUDA_VISIBLE_DEVICES="..." python \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="./checkpoints" \
    --audioset-dir="./data/audioset_small/audiofolder" \
    --audiocaps-dir="./data/audiocaps/audiofolder" \
    --training-config="./configs/pretrain_1on1_large_config.yaml" \
    --wandb-group="pretraining"
```

Argument `--training-config` is the most important - it specifies everything important about training.
We experimented with different setups. you can find the different configs inside `configs/` folder.


### Finetuning

To run finetuning, use the following command:

```shell
CUDA_VISIBLE_DEVICES="..." python \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="./checkpoints" \
    --clotho-dir="./data/clotho_v2.1/audiofolder" \
    --training-config="./configs/finetune_large_config.yaml" \
    --wandb-group="finetuning"
```

TODO make it so that it can load a pretraing checkpoint from local file.



## Multitask training and inference

To effectively train on multiple datasets, we put a dataset and task identifiers into the captions.

Example:
- **clotho > caption:** Fair kind music is being played at the circus grounds.
- **audiocaps > caption:** The wind is blowing, insects are singing, and rustling occurs
- **audioset > keywords:** boat - water vehicle, motorboat - speedboat, sounds of things, vehicle

The prefix informs the model about the style of caption that is used. During inference, a prefix is
forced to the decoder, which makes the model generate output in a desired style. This is a trick
inspired by multilingual generative language models where the prefix specifies the output language.


## Licence

For all code in this repository code, licence in LICENSE file applies.
For the files in the `data` directory, specific licences apply: 

- AudioSet labels: CC BY 4.0
  - source of data: <https://research.google.com/audioset/>
- AudioSet ontology: CC BY-SA 4.0
  - source of data: <https://research.google.com/audioset/>
- AudioCaps labels: MIT
  - source of data: <https://github.com/cdjkim/audiocaps>
