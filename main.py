import warnings; warnings.filterwarnings("ignore")
import argparse
import dotenv; dotenv.load_dotenv()
import os
import uuid
import pandas as pd
import soundfile as sf
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from minio import Minio
from omnivoice import OmniVoice


def worker(rank, indices_list, texts, references_records, N, output_dir, upload_dir=None, delete_local=False):
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

    client = Minio(
        "lake-api.actable.ai:9000",
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False
    )
    for idx in tqdm(my_indices, desc=f"GPU{rank}", position=rank):
        row = references_records[idx]
        for jdx in range(N):
            audio_id = str(uuid.uuid4())
            text = texts[idx * N + jdx]
            audio = model.generate(
                text=text,
                ref_audio=row["filepath"],
                ref_text=row['text'],
            )
            audio_path = os.path.join(audios_dir, f"{audio_id}.wav")
            metadata_path = os.path.join(metas_dir, f"{audio_id}.txt")
            sf.write(audio_path, audio[0], 24000)
            with open(metadata_path, "w") as f:
                f.write(f"{text}\n{row['filepath']}\n{row['speaker_id']}")

            if upload_dir is not None:
                bucket = upload_dir.split('/')[0]
                path = '/'.join(upload_dir.split('/')[1:])
                audio_upload = os.path.join(path, f"audios/{audio_id}.wav")
                metadata_upload = os.path.join(path, f"metadatas/{audio_id}.txt")
                try:
                    client.fput_object(bucket, audio_upload, audio_path)
                    client.fput_object(bucket, metadata_upload, metadata_path)
                except Exception as e:
                    print(f"Error occurred while uploading {audio_id}: {e}")
                if delete_local:    
                    os.remove(audio_path)
                    os.remove(metadata_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-R", type=int, default=1000, help="Value of R")
    parser.add_argument("-P", type=int, default=0, help="Value of P")
    parser.add_argument("-N", type=int, default=100, help="Value of N")
    parser.add_argument("-u", type=str, default="dataset", help="Upload Dataset on Minio or not")
    parser.add_argument("-d", action="store_true", help="Delete local files after uploading to Minio")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    R = args.R
    P = args.P
    N = args.N

    print(R, P, N)

    df_texts = pd.read_csv("metadatas/PhoMT_training.csv")["vi"]
    print(df_texts.nunique())
    texts = df_texts.tolist()

    df_references = pd.read_csv("metadatas/single_speakers.csv")
    print(df_references.nunique())

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
        args=(indices_list, texts, references_records, N, "dataset", args.u, args.d),
        nprocs=num_gpus,
        join=True,
    )

    metadata_paths = [os.path.join("dataset/metadatas", path) for path in os.listdir("dataset/metadatas")]
    metadatas = []
    for metadata_path in tqdm(metadata_paths):
        with open(metadata_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines[1] = lines[1].split("ViMD/", 1)[-1]
            lines[1] = "ViMD/" + lines[1]
            audio_id = os.path.basename(metadata_path).split('.')[0]
            audio_path = os.path.join("audios", f"{audio_id}.wav")
            lines.insert(0, audio_path)
            metadatas.append(lines)
    metadata_df = pd.DataFrame(metadatas, columns=['filepath', 'text', 'reference', 'speaker_id'])
    metadata_df.to_csv("dataset/metadata.csv", index=False)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()