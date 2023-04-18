import os
import logging
import pandas as pd
import time
import glob
from tqdm import tqdm
import typer
import pathlib
import math
from typing import Optional, List, Union
from multiprocessing import Process, Lock

app = typer.Typer()

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na

def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging

def df_to_n_chunks(df, n) -> List[pd.DataFrame]:
    chunk_size = len(df) // n + 1
    dfs = []
    for i in range(n):
        new_df = df.iloc[i*chunk_size:(i+1)*chunk_size]
        dfs.append(new_df)
    return dfs

def download(df: pd.DataFrame,
             audios_dir: pathlib.Path,
             logging,
             logs_dir,
             pid: Optional[int] = 0, 
             lock: Optional[Lock] = None):
    """
    download yt videos specified in df, can be used in a multiprocess manner,
    just specify lock parameter, so logging goes safe
    """
    print(f"### Download process number {pid} started")
    for row in tqdm(df.iterrows()):
        n = row[0]
        audio_id = row[1]["audiocap_id"]
        youtube_id = row[1]["youtube_id"]
        start_time = row[1]["start_time"]
        end_time = start_time + 10
        duration = 10

        # Download full video of whatever format
        audio_name = os.path.join(audios_dir, f'{audio_id}_{youtube_id}')
        success = os.system(f"yt-dlp -S "asr:32000" -x --quiet --audio-format mp3 --external-downloader aria2c --external-downloader-args 'ffmpeg_i:-ss {start_time} -to {end_time}' -o '{audio_name}.%(ext)s' https://www.youtube.com/watch?v={youtube_id}")

        if lock: # logging, if multiprocessing, use lock
            with lock:
                if success == 0: # successfully downloaded
                    logging.info('{} {} start_time: {:.1f}, end_time: {:.1f}'.format(
                        n, audio_id, start_time, end_time))
                else:  # log unsuccessful audio_id for future filtering
                    print(f"logging {audio_id} as unsuccessful into {logs_dir}/unsuccessful_ids.log")
                    os.system(f"echo '{audio_id}' >> {logs_dir}/unsuccessful_ids.log")
        else:
            if success == 0:  # successfully downloaded
                logging.info('{} {} start_time: {:.1f}, end_time: {:.1f}'.format(
                        n, audio_id, start_time, end_time))
            else:   # log unsuccessful audio_id for future filtering
                print(f"logging {audio_id} as unsuccessful into {logs_dir}/unsuccessful_ids.log")
                os.system(f"echo '{audio_id}' >> {logs_dir}/unsuccessful_ids.log")


@app.command()
def download_audiocaps_wavs(
    mini_data: Optional[bool] = typer.Option(False, "-m", "--mini", help="Whether or not to download only first 10 videos"),
    from_id: Optional[int] = typer.Option(0, "-f", "--from", help="From which audiocap_id to start the downloading (excluding)"),
    to_id: Optional[float] = typer.Option(math.inf, "-t", "--to", help="To which audiocap_id to start the downloading (including)"), 
    num_workers: Optional[int] = typer.Option(1, "-w", "--num-workers", help="Number of processes to use for downloading."), 
    csv_path: pathlib.Path = typer.Argument(..., help="Path to the audioCap csv"),
    audios_dir: pathlib.Path = typer.Argument(..., help="Path where to save audios"),
    ):
    
    """
    Download audiocaps dataset using the csvs obtained at https://github.com/cdjkim/audiocaps/tree/master/dataset
    
    RIGHT NOW the app is run by:
    AUDIOCAPS_DIR=./data/audiocaps
    python audiocap/download_audiocaps.py -w 2 -f 95000 -t 100000 $AUDIOCAPS_DIR"/csvs/debug.csv" $AUDIOCAPS_DIR"/audios/debug_set"
    
    Some files don't get fully downloaded - they're logged into unsuccessful_ids.log, but leave a '*.mp4.part' file behind.
    To get rid of them use: `find . -name '*.part' -exec rm {} \;` bash command.
    """
    assert from_id <= to_id, "Option FROM should be smaller or equal than option TO" # input check

    if mini_data:
        logs_dir = f'{audios_dir}/../_logs/download_dataset/{get_filename(csv_path)}'
    else:
        logs_dir = f'{audios_dir}/../_logs/download_dataset_minidata/{get_filename(csv_path)}'

    print(f"### CREATING FOLDERS {audios_dir} and {logs_dir}")
    create_folder(audios_dir)
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Download log is saved to {}'.format(logs_dir))

    # Read csv
    print("### LOADING CSV")
    df = pd.read_csv(csv_path)
    # sort so we download increasingly by audio_id and can easily resume
    df.sort_values(by=["audiocap_id"], inplace=True) 
    df = df[(df["audiocap_id"] > from_id) & (df["audiocap_id"] <= to_id)]
        
    if mini_data:
        df = df.head(10)  # Download partial data for debug
    
    logging.info(f"### Num of videos to download: {len(df)}")
    logging.info(f"### Num of workers: {num_workers}")
    download_start_time = time.time()

    # init multiprocessing
    if num_workers > 1:
        df_chunks = df_to_n_chunks(df, num_workers)
        lock = Lock()
        processes = [Process(target=download, args=(df_chunks[i], audios_dir, logging, logs_dir, i, lock)) for i in range(num_workers)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
    else: download(df, audios_dir, logging, logs_dir)
                        
            
    print(f'### Download finished! The whole thing took {time.time() - download_start_time :.3f} s')
    

if __name__ == "__main__":
    app()
