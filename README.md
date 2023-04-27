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



### Getting the Clotho dataset

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

### Create Clotho AudioFolder

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


### Getting AudioSet dataset

Make the script executable

```shell
chmod +x ./scripts/download_audioset.sh
```

Download the audio files

```shell
SPLIT='train_unbalanced' # run again with 'train_balanced' or 'eval'

mkdir -p logs/download_audioset

./scripts/download_audioset.sh \
    "data/audioset/csvs/${SPLIT}.csv" \
    "data/audioset/audios/${SPLIT}/" 2>&1 \
    | tee >( sed 's/.*\r//' > "logs/download_audioset/${SPLIT}.txt" )
```

(`sed` is there to delete output lines that just update the progress)


### Getting AudioCaps dataset

TODO


### Create a balanced AudioSet subset

TODO


### Create AudioSet small AudioFolder

TODO

### Create AudioCaps AudioFolder

TODO


## Pretraining 

```shell
CUDA_VISIBLE_DEVICES="..." python \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="./checkpoints" \
    --audioset-dir="./data/audioset_small/audiofolder" \
    --audiocaps-dir="./data/audiocaps/audiofolder" \
    --training-config="./configs/pretrain_1on1_large_config.yaml" \
    --wandb-group="pretraining"
```


## Finetuning

```shell
CUDA_VISIBLE_DEVICES="..." python \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="./checkpoints" \
    --clotho-dir="./data/clotho_v2.1/audiofolder" \
    --training-config="./configs/finetune_large_config.yaml" \
    --wandb-group="finetuning"
```

## Licence

For all code in this repository code, licence in LICENSE file applies.
For the files in the `data` directory, specific licences apply: 

- AudioSet labels: CC BY 4.0
  - source of data: <https://research.google.com/audioset/>
- AudioSet ontology: CC BY-SA 4.0
  - source of data: <https://research.google.com/audioset/>
- AudioCaps labels: MIT
  - source of data: <https://github.com/cdjkim/audiocaps>
