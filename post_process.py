import os
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", "-bp", type=str, default="dataset", help="Base path")
    args = parser.parse_args()

    metadata_paths = [os.path.join(args.base_path, "metadatas", path) for path in os.listdir(os.path.join(args.base_path, "metadatas"))]
    metadatas = []
    for metadata_path in metadata_paths:
        with open(metadata_path, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines[1] = lines[1].split("ViMD/", 1)[-1]
            lines[1] = "ViMD/" + lines[1]
            audio_id, _ = os.path.splitext(metadata_path)
            audio_path = os.path.join("audios", f"{audio_id}.wav")
            lines.insert(0, audio_path)
            metadatas.append(lines)
    metadata_df = pd.DataFrame(metadatas, columns=['filepath', 'text', 'reference', 'speaker_id'])
    metadata_df.to_csv(os.path.join(args.base_path, "metadata.csv"), index=False)
    pass