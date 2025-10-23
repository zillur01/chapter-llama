!/bin/bash

python tools/download/models.py "captions_asr-1k" --local_dir "."
mkdir dataset
bash tools/download/captions.sh asr_s10k-2_train_preds+no-asr-10s
bash tools/download/docs.sh sml1k_train
bash tools/download/docs.sh sml300_val
bash tools/download/docs.sh s100_val
