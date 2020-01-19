#data process
#---insert wiki-cross and yago dictionary into mongodb
python entity/mongo_utils.py

#change 'Project Absolute Path' into your project pathes
#---generate candidates
python s2_1_get_mention_cands.py
python s2_2_get_e_mention_cands.py

#---check our generated candidates with EMNLP2017
python s3_1_check_entity_cands.py

#---get all entities
python s3_2_get_entity_2_id.py

#---extract entity Ganea embeddings, transE embedding, word2vec embedding
python s3_3_get_entity_2_embed.py

#---extract entity notable type from freebase
python s3_4_get_entity_type.py

#---extract all words embedding
python steps/s4_gen_word_infos.py

#insert training data into the mongodb
python embeddings/get_ent_linking_data.py

#---------------------------
#generate elmo presentation for mention surface name feature...
#--------------------------------------
python train_global_gen_elmo.py --test=False --data_source=EntityLinking --margin_param=0.1 --learning_rate_start=1e-4 --loss_type=margin --model_type=SimpleCNNLocalEntLinkModel --epoch=150 --lbp_iter_num=2 --keep_prob=0.8 --WS_weight=WS --gcn_activation=tanh --gcn_hidden_V=True --l2_w=0.0 --reg_w=0.0 --use_unk=none --keep_prob_V=1.0 --residual_w=1.0  --keep_prob_D=0.5 --A_diag=dense --start_num=0 --re_train_num=0  --optimizer=normal --cand_nums=30 --data_tag_number=0 --data_item_number=0

#training local model
bash local.sh

#re-prune candidates
bash get_repruned_cand.sh

#get pre-trained context score
bash get_pretrained_ctx_score.sh

#training global model
bash global.sh
