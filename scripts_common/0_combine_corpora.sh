#!/usr/bin/env bash

# Source directories
bbc_dir=/Users/caiwingfield/corpora/BBC/3\ Replaced\ symbols
bnc_speech_dir=/Users/caiwingfield/corpora/BNC/1\ Speech
bnc_text_dir=/Users/caiwingfield/corpora/BNC/1\ Text
bnc_text_for_speech_dir=/Users/caiwingfield/corpora/BNC/1\ Text\ for\ speech

# Target directories
combined_speech_dir=/Users/caiwingfield/corpora/Combined/0\ SPEECH
combined_text_dir=/Users/caiwingfield/corpora/Combined/0\ TEXT

# Have to do it in loops because there are too many files for cp to handle them all at once

echo "Copying BBC subtitles to speech corpus"
for f in "$bbc_dir"/* ; do
    cp "$f" "$combined_speech_dir"
done
echo "Copying BNC speech to speech corpus"
for f in "$bnc_speech_dir"/* ; do
    cp "$f" "$combined_speech_dir"
done
echo "Copying BNC text-for-speech to speech corpus"
for f in "$bnc_text_for_speech_dir"/* ; do
    cp "$f" "$combined_speech_dir"
done

echo "Copying BNC text to text corpus"
for f in "$bnc_text_dir"/* ; do
    cp "$f" "$combined_text_dir"
done
