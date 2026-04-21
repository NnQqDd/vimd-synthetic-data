import dotenv; dotenv.load_dotenv()
import os
from huggingface_hub import hf_hub_download

os.makedirs("metadatas", exist_ok=True)

_ = hf_hub_download(
    repo_id="ura-hcmut/PhoMT",
    repo_type='dataset',
    filename='PhoMT_training.csv',
    local_dir="metadatas",
    # token = '',
)
