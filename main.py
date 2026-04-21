import os
import uuid
from tqdm import tqdm
import pandas as pd
from omnivoice import OmniVoice
import soundfile as sf
import torch

# Load the model
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)


df_texts = pd.read_csv("metadatas/PhoMT_training.csv")['vi']
print(df_texts.nunique())
texts = df_texts.tolist()

df_references = pd.read_csv("metadatas/single_speakers.csv")
print(df_references.nunique())

R = 1000
P = 0
N = 100

print(R, P, N)

os.makedirs("dataset/audios", exist_ok=True)
os.makedirs("dataset/metadatas", exist_ok=True)
for idx, row in tqdm(df_references.iterrows(), total=min(len(df_references), R + P), desc="Speaker"):
    if idx < P:
        continue
    for jdx in tqdm(range(N), desc="Audio"):
        audio_id = str(uuid.uuid4())
        audio = model.generate(
            text=texts[idx*N + jdx],
            ref_audio=row['filepath'],
            # ref_text="Transcription of the reference audio.",
        ) 
        
        sf.write(os.path.join("dataset", "audios", f"{audio_id}.wav"), audio[0], 24000)

        with open(os.path.join("dataset", "metadatas", f"{audio_id}.txt"), "w") as f:
            f.write(f"{texts[idx]}\n{row['filepath']}\n{row['speaker_id']}")

        