for date in 7_15
do
{
  for start_num in 0
  do
  {
    for has_type in T
    do
    {
      for feature in S012
      do
      {
        for s2_width in 5
        do
        {
          for s2_width_0_1 in 5
          do
          {
            for margin_param in 0.01
            do
            {
              for keep_prob in 0.8
              do
              {
                for s2_width_0_elmo in 20
                do
                {
                  CUDA_VISIBLE_DEVICES=0 python train_local_EL_2.py --test=True --margin_param=${margin_param} --learning_rate_start=1e-4 --logs_path=logs/CNN2 --loss_type=margin --model_type=SimpleCtxCNNLocalEntLinkModel --iter_num=0 --epoch=50 --re_train_num=1 --start_num=${start_num} --max_norm=0.0 --s2_width_0=${s2_width} --s2_width_0_1=${s2_width_0_1} --s2_width_1=${s2_width} --s2_width_2=${s2_width} --keep_prob=${keep_prob} --restore=checkpoint_local/${date} --has_type=${has_type} --feature=${feature} --use_unk=none --date=${date} --seed=721818695 --s2_width_0_elmo=${s2_width_0_elmo}  --insert_sub=True --data_source=EntityLinking
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
