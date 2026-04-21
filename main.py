import warnings; warnings.filterwarnings("ignore")

import os
import uuid
import pandas as pd
import soundfile as sf
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from omnivoice import OmniVoice


def worker(rank, indices_list, texts, references_records, N, output_dir):
    # rank is automatically passed by mp.spawn as the process index → use it as GPU id
    device = f"cuda:{rank}"
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=torch.float16,
    )

    audios_dir = os.path.join(output_dir, "audios")
    metas_dir = os.path.join(output_dir, "metadatas")

    my_indices = indices_list[rank]
    for idx in tqdm(my_indices, desc=f"GPU{rank}", position=rank):
        row = references_records[idx]
        for jdx in range(N):
            audio_id = str(uuid.uuid4())
            audio = model.generate(
                text=texts[idx * N + jdx],
                ref_audio=row["filepath"],
                # ref_text="Transcription of the reference audio.",
            )
            sf.write(os.path.join(audios_dir, f"{audio_id}.wav"), audio[0], 24000)
            with open(os.path.join(metas_dir, f"{audio_id}.txt"), "w") as f:
                f.write(f"{texts[idx]}\n{row['filepath']}\n{row['speaker_id']}")


def main():
    df_texts = pd.read_csv("metadatas/PhoMT_training.csv")["vi"]
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

    end = min(len(df_references), R + P)
    all_indices = list(range(P, end))

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA GPUs available"
    print(f"Using {num_gpus} GPUs")

    # Round-robin split so GPUs finish around the same time
    indices_list = [all_indices[i::num_gpus] for i in range(num_gpus)]

    # Pass plain python objects instead of a DataFrame (cleaner pickling)
    references_records = df_references.to_dict("records")

    mp.spawn(
        worker,
        args=(indices_list, texts, references_records, N, "dataset"),
        nprocs=num_gpus,
        join=True,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()