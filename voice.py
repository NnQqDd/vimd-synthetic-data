import warnings; warnings.filterwarnings("ignore")
import argparse
import dotenv; dotenv.load_dotenv()
import os
import random
import uuid
import pandas as pd
import soundfile as sf
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from minio import Minio
from omnivoice import OmniVoice

GENDERS = ["男", "女"]
AGES = ["儿童", "少年", "青年", "中年", "老年"]
PITCHES = ["极低音调", "低音调", "中音调", "高音调", "极高音调"]
STYLE = ["耳语"]
DIALECTS = ["河南话", "陕西话", "四川话", "贵州话", "云南话", "桂林话", "济南话", "石家庄话", "甘肃话", "宁夏话", "青岛话", "东北话"]


def random_instruct(seed):
    random.seed(seed)
    if random.random() < 0.5:
        return None
    gender = random.choice(GENDERS)
    # age = random.choice(AGES)
    # pitch = random.choice(PITCHES)
    # style = random.choice(STYLE)
    dialect = random.choice(DIALECTS)
    return f"{gender}, {dialect}"


def worker(rank, indices_list, texts, output_dir, upload_dir=None, delete_local=False):
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
        os.getenv("MINIO_URL"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False
    )
    for idx in tqdm(my_indices, desc=f"GPU{rank}", position=rank):
        audio_id = str(uuid.uuid4())
        text = texts[idx]
        # instruct = random_instruct(42)
        audio = model.generate(
            text=text,
            # instruct=instruct,
        )
        audio_path = os.path.join(audios_dir, f"{audio_id}.wav")
        metadata_path = os.path.join(metas_dir, f"{audio_id}.txt")
        sf.write(audio_path, audio[0], 24000)
        with open(metadata_path, "w") as f:
            f.write(f"{text}")

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
    parser.add_argument("-N", type=int, default=100, help="Value of N")
    parser.add_argument("-P", type=int, default=0, help="Value of P")
    parser.add_argument("-u", type=str, help="Upload Dataset on Minio or not")
    parser.add_argument("-d", action="store_true", help="Delete local files after uploading to Minio")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    N = args.N
    P = args.P
    print(N, P)

    df_texts = pd.read_csv("metadatas/PhoMT_training.csv")["vi"]
    print(df_texts.nunique())
    texts = df_texts.tolist()

    os.makedirs("dataset/audios", exist_ok=True)
    os.makedirs("dataset/metadatas", exist_ok=True)

    all_indices = list(range(P, N + P))

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA GPUs available"
    print(f"Using {num_gpus} GPUs")

    # Round-robin split so GPUs finish around the same time
    indices_list = [all_indices[i::num_gpus] for i in range(num_gpus)]

    mp.spawn(
        worker,
        args=(indices_list, texts, "dataset", args.u, args.d),
        nprocs=num_gpus,
        join=True,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()