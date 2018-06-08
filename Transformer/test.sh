CUDA_VISIBLE_DEVICES=4 python main.py translate --model save_model/nmt.autosave.pkl.10000 --corpus /home/weixiangpeng/NMT/wmt14/ende/corpus/test.en --translation bleu/translation.de --beamsize 10
