seed=721818695
data_source=GCN_EL_june_3_5
s2_width_0=5
sim_scale=1.0
for margin_param in 0.1
do
{
  for date in 7_15
  do
  {
    for s2_width_0_elmo in 20
    do
    {
      for diag_self in 0.5
      do
      {
        for A_adj_mask in unmask
        do
        {
          for A_diag_dim in 40
          do
          {
            for mask_type in dist_count
            do
            {
              for lbp_iter_num in 7
              do
              {
                for gcn_kernel_size in 1
                do
                {
                  for start_num in 0
                  do
                  {
                    CUDA_VISIBLE_DEVICES=0 python train_global_EL.py --test=True --data_source=${data_source} --margin_param=${margin_param} --learning_rate_start=1e-4 --loss_type=margin --model_type=RDGraphCNNGlobalEntLinkModel --epoch=100 --lbp_iter_num=${lbp_iter_num} --seed=${seed} --s2_width_0_elmo=${s2_width_0_elmo} --keep_prob=0.8 --A_adj_mask=${A_adj_mask} --restore=checkpoint_np_global/${data_source}/${date} --WS_weight=WS --gcn_activation=relu --gcn_hidden_V=True --l2_w=0.0 --reg_w=0.0 --A_diag_dim=${A_diag_dim} --keep_prob_D=0.8 --date=${date} --diag_self=${diag_self} --A_diag=dense --start_num=${start_num} --re_train_num=0 --cand_nums=5 --message_opt=max --mask_type=${mask_type} --gcn_kernel_size=${gcn_kernel_size} --s2_width_1_elmo=1 --s2_width_0_1=5 --s2_width_0=${s2_width_0} --s2_width_1=${s2_width_0} --s2_width_2=${s2_width_0} --residual_w=1.0 --sim_scale=${sim_scale} --score_merge_type=MLP
                  }
                  done
                }
                done
              }
              done
            }
            done
          }
          done
        }
        done
      }
      done
    }
    done
  }
  done
}
done
