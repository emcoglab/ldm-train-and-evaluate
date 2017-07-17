#!/usr/bin/env bash

# Source directories
bbc_dir=/Users/caiwingfield/Langboot\ local/Corpora/BBC/3\ Replaced\ symbols
bnc_speech_dir=/Users/caiwingfield/Langboot\ local/Corpora/BNC/1\ Speech
bnc_text_dir=/Users/caiwingfield/Langboot\ local/Corpora/BNC/1\ Text
bnc_text_for_speech_dir=/Users/caiwingfield/Langboot\ local/Corpora/BNC/1\ Text\ for\ speech

# Target directories
combined_speech_dir=/Users/caiwingfield/Langboot\ local/Corpora/Combined/0\ SPEECH
combined_text_dir=/Users/caiwingfield/Langboot\ local/Corpora/Combined/0\ TEXT

echo "========"
echo "Copying speech documents"
echo "========"
echo "Copying BBC subtitles"
for f in "$bbc_dir"/* ; do
    cp "$f" "$combined_speech_dir"
done
echo "--------"
echo "Copying BNC speech"
for f in "$bnc_speech_dir"/* ; do
    cp "$f" "$combined_speech_dir"
done
echo "--------"
echo "Copying BNC text-for-speech"
for f in "$bnc_text_for_speech_dir"/* ; do
    cp "$f" "$combined_speech_dir"
done

echo "========"
echo "Copying text documents"
echo "========"
echo "Copying BNC text"
for f in "$bnc_text_dir"/* ; do
    cp "$f" "$combined_text_dir"
done
