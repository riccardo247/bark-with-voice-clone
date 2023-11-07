#!/bin/bash

#this set up all the repo and env to do the fine tuning, including download data from bb get file list and encoding it
# Exit script if any command fails
set -e

# Create the virtual environment in /ve/bark
mkdir -p /ve/bark
python3 -m venv /ve/bark

# Activate the virtual environment
source /ve/bark/bin/activate

# Install libraries with pip install
pip install accelerate torchaudio git+https://github.com/huggingface/transformers.git
pip install --upgrade "diffusers[torch]"
pip install funcy encoded

mkdir /finetune
# Clone the required repositories
git clone https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer /finetune/hubert
pip install -r /finetune/hubert/requirements.txt

git clone https://github.com/riccardo247/bark-with-voice-clone /finetune/bark_with_voice_clone

git clone https://github.com/suno-ai/bark /finetune/bark

# Download and install B2 Command Line Tool
wget -P /finetune/ https://github.com/Backblaze/B2_Command_Line_Tool/releases/download/v3.11.0/b2-3.11.0.tar.gz 
tar -xzvf /finetune/b2-3.11.0.tar.gz
cd /finetune/b2-3.11.0
python3 setup.py install

# Authorize B2 account
b2 authorize-account "$BB_ID" "$BB_KEY"

# Sync the bucket to the local directory
b2 sync b2://78ba729040ff95f08cb80612 /finetune/audio

# Go back to the root directory 
cd /finetune
git clone https://github.com/riccardo247/SALMONN

#
#get list of files was, text
python3 /finetune/SALMONN/get_files_list.py
#encode files
python3 /finetune/bark_with_voice_clone/encode_files.py

# Deactivate the virtual environment
deactivate
