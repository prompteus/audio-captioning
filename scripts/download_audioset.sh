#!/bin/bash

download_yt() {
    local line=${1}
    local folder=${2}
    local extention=${3}

    local yt_id=$(echo $line | cut -d ',' -f1)
    local start_seconds=$(echo $line | cut -d ',' -f2)
    local end_seconds=$(echo $line | cut -d ',' -f3)

    local filepath="$folder/$yt_id.$extention"
    local url="https://www.youtube.com/watch?v=$yt_id"

    if [ -f filepath ]; then
        return
    fi

    yt-dlp \
        -S "asr:32000" \
        -x \
        -o "${filepath}" \
        --quiet \
        --no-progress \
        --extract-audio \
        --audio-format mp3 \
        --external-downloader aria2c \
        --downloader ffmpeg \
        --downloader-args "ffmpeg_i:-ss ${start_seconds} -to ${end_seconds}" \
        "${url}"
}

export -f download_yt

csv_file=$1
folder=$2

nice -n 19 \
    cat "${csv_file}" \
    | grep -v '^#' \
    | parallel --progress "download_yt '{}' '${folder}' mp3 || true" 
