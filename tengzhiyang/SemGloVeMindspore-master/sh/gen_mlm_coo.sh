CUDA_VISIBLE_DEVICES=2 python src/bert_cooccur_mindspore.py \
--corpus_name sample \
--corpus_path /data/home/ganleilei/BertGloVe/wiki/ \
--model_name bert-base-uncased \
--bert_path /home/ganleilei/data/bert/ \
--save_path /data/home/ganleilei/BertGloVe/wiki/ \
--divide \
--window_size 20 \
--mlm_glove \