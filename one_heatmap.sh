file="/cluster/home/pokanovic/gdu/results/"
#method="cosine_similarity"
test_source="mnist"
for method in "cosine_similarity" "MMD" "projected"
do
  for lambda_sparse in 0 0.1 0.01 0.001
  do
    for lambda_OLS in 0 0.1 0.01 0.001
    do
        bsub -W 20:00 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" "python SimulationExperiments/experiment_4_digits/digits_5_classification.py --run_all 0 --res_file_dir $file  --method $method --TARGET_DOMAIN $test_source --fine_tune False --lambda_orth 0 --lambda_sparse $lambda_sparse --lambda_OLS $lambda_OLS >& one_heatmap${test_source}_${method}_e2e_${lambda_sparse}_${lambda_OLS}.out"
    done
  done
done

