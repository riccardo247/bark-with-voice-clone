from bark_hubert_quantizer.hubert_manager import HuBERTManager
from bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from bark_hubert_quantizer.customtokenizer import CustomTokenizer
from encodec.utils import convert_audio
import torchaudio
import os
import numpy as np
import torch

max_duration_sec = 15.1 # the maximum allowed duration in seconds
device = "cuda"
path = "/finetune/data"
SAMPLE_RATE = 24_000
CHANNELS = 1

hubert_manager = HuBERTManager()
hubert_manager.make_sure_hubert_installed()
hubert_manager.make_sure_tokenizer_installed()

hubert_path = '/finetune/data/models/hubert/hubert.pt'
hubert_tokenizer_path = '/finetune/data/models/hubert/tokenizer.pth'

# Load the HuBERT model
hubert_model = CustomHubert(checkpoint_path=hubert_path).to(device)
hubert_model.eval()
for param in hubert_model.parameters():
    param.requires_grad = False

# Load the CustomTokenizer model
hubert_tokenizer = CustomTokenizer.load_from_checkpoint(hubert_tokenizer_path).to(device)  # Automatically uses the right layers

from bark.generation import load_codec_model
codec_model = load_codec_model(use_gpu=True)
codec_model.eval()
for param in codec_model.parameters():
    param.requires_grad = False


def get_duration(wav, sr):
    return wav.shape[1] / sr

def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8', errors='ignore') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
        #base = os.path.dirname(filename)
        #for j in range(len(filepaths_and_text)):
        #    filepaths_and_text[j][0] = os.path.join(base, filepaths_and_text[j][0])
    return filepaths_and_text

def main():
  valid_lines_train = []
  # convert wavs to semantic tokens
  for wav_path, txt ,dir_path in load_filepaths_and_text("/finetune/file_list.txt"):
      print(f"{wav_path}")
      wav, sr = torchaudio.load(wav_path)
      if not get_duration(wav, sr) > max_duration_sec:
          valid_lines_train.append((wav_path, txt))
      wav = convert_audio(wav, sr, SAMPLE_RATE, CHANNELS).to(device)
  
      semantic_vectors = hubert_model.forward(wav, input_sample_hz=SAMPLE_RATE)
      semantic_tokens = hubert_tokenizer.get_token(semantic_vectors)
  
      # save semantic tokens
      os.makedirs(os.path.join(path, 'tokens'), exist_ok=True)
      semantic_tokens = semantic_tokens.cpu().numpy()
  
      # Extract discrete codes from EnCodec
      with torch.no_grad():
          encoded_frames = codec_model.encode(wav.unsqueeze(0))
      codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
  
      # move codes to cpu
      codes = codes.cpu().numpy()
  
      #create name
      #get chunk name and file as well
      base_name = os.path.basename(wav_path)
      orig_file =  dir_path.split(os.path.sep)[-1]
      filename = base_name
      filename = filename.replace('.wav', '.npz')
      # save tokens
      token_path = dir_path+"_tokens"
      os.makedirs(token_path, exist_ok=True)
      print(f"{os.path.join(token_path,  filename)}")
      np.savez_compressed(os.path.join(token_path,  filename), fine=codes, coarse=codes[:2, :], semantic=semantic_tokens)

if __name__ == '__main__':
  main()
