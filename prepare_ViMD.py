import os
import pandas as pd


def prepare_speech_metadata(metadata_path):
    metadata_df = pd.read_csv(metadata_path)
        
    filename_column, speaker_id_column = None, None
    for column_name in metadata_df.columns:
        lower_name = column_name.lower()
        if 'file' in lower_name and ('name' in lower_name or 'path' in lower_name):
            filename_column = column_name
        if 'speaker' in lower_name:
            speaker_id_column = column_name
        
    if filename_column is None or speaker_id_column is None:
        raise Exception(f'"{metadata_path}" does not satisfy the condition in the YAML file!')
        
    metadata_df.rename(columns={
        filename_column: 'filepath',
        speaker_id_column: 'speaker_id',
    }, inplace=True)
    
    abs_path = os.path.dirname(os.path.abspath(metadata_path))
    metadata_df['filepath'] = metadata_df['filepath'].apply(
        lambda p: os.path.join(abs_path, p)
        if isinstance(p, str) and (len(p) == 0 or p[0] != '/') else p
    )
    
    if 'split' in metadata_df.columns.tolist():
        metadata_df.rename(columns={
            'split': 'partition'
        }, inplace=True)

    return metadata_df


if __name__ == '__main__':
    METADATA_PATH = '/home/duyn/ActableDuy/datasets/ViMD/metadata.csv'
    metadata_df = prepare_speech_metadata(METADATA_PATH)

    speaker_counts = metadata_df['speaker_id'].value_counts()

    # Map counts back to dataframe
    metadata_df['num_audios'] = metadata_df['speaker_id'].map(speaker_counts)

    # Split
    df_single = metadata_df[metadata_df['num_audios'] == 1]
    df_multi  = metadata_df[metadata_df['num_audios'] >= 2]

    # (Optional) drop helper column
    df_single = df_single.drop(columns=['num_audios'])
    df_multi  = df_multi.drop(columns=['num_audios'])

    print(df_single)
    print(df_multi)

    df_single.to_csv("metadatas/single_speakers.csv", index=False)
    df_multi.to_csv("metadatas/multi_speakers.csv", index=False)