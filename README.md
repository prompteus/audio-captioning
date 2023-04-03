# Audio captioning


## Setting up environment

Start by creating a conda environment:
```shell
git clone ...
cd audio-captioning
conda create -n malach23 python=3.8
conda activate malach23
pip install -r requirements.txt
pip install -e .
```
If the last line does not work, update your pip. e.g. `pip install --upgrade pip`



## Getting the Clotho dataset

```shell
mkdir -p data/clotho_v2.1
```

Download the data from <https://zenodo.org/record/4783391> and extract it into the data/clotho_v2.1 folder. Your tree structure should look like this

```
audio-captioning/
├── audiocap
│   ...
...
|
├── data
│   └── clotho_v2.1
│       ├── clotho_captions_development.csv
│       ├── clotho_captions_evaluation.csv
│       ├── clotho_captions_validation.csv
│       ├── clotho_metadata_development.csv
│       ├── clotho_metadata_evaluation.csv
│       ├── clotho_metadata_test.csv
│       ├── clotho_metadata_validation.csv
│       ├── development
│       ├── evaluation
│       ├── test
│       └── validation
...
```

Now, run 

```shell
python audiocap/prepare_audiofolder.py data/clotho_v2.1/
```

This will prepare the folder into the format that is easily loadable by Huggingface datasets library: `datasets.load_dataset("audiofolder", ...)`


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

To try out the training notebook, go to `notebooks/train_whisper_supervised.ipynb`

During training, the loss, metrics and example predictions are logged to `wandb`.
