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



## Getting the Clotho dataset

```shell
mkdir -p data/clotho_v2.1/audiofolder
```

Download the data from <https://zenodo.org/record/4783391> and extract csv into the `data/clotho_v2.1` and audios into `data/clotho_v2.1/audiofolder` folder. Your tree structure should look like this

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
│       │   ├─ validation
│       ├── clotho_captions_development.csv
│       ├── clotho_captions_evaluation.csv
│       ├── clotho_captions_validation.csv
│       ├── clotho_metadata_development.csv
│       ├── clotho_metadata_evaluation.csv
│       ├── clotho_metadata_test.csv
│       └── clotho_metadata_validation.csv
...
```

Now, run 

```shell
python audiocap/prepare_audiofolder.py data/clotho_v2.1/
```

This will prepare the folder into the format that is easily loadable.

To limit a size of a split (like validation and evaluation), run:

```shell
python audiocap/prepare_audiofolder.py limit-clotho-split data/clotho_v2.1/audiofolder/ validation --limit 200
python audiocap/prepare_audiofolder.py limit-clotho-split data/clotho_v2.1/audiofolder/ evaluation --limit 400
```

This will sample (with a seed) a subset with a desired size and move the remaining examples to the development split.


## Getting Audioset dataset

Make the script executable

```shell
chmod +x ./scripts/download_audioset.sh
```

Download the audio files

```shell
SPLIT='train_unbalanced' # or 'train_balanced' or 'eval'

mkdir -p logs/download_audioset

./scripts/download_audioset.sh \
    "data/audioset/csvs/${SPLIT}.csv" \
    "data/audioset/audios/${SPLIT}/" 2>&1 \
    | tee >( sed 's/.*\r//' > "logs/download_audioset/${SPLIT}.txt" )
```

(`sed` is there to delete output lines that just update the progress)



## Training

To try out the training notebook, go to `audiocap/train_whisper_supervised.py`

During training, the loss, metrics and example predictions are logged to `wandb`.
