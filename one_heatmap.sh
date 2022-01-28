file="/cluster/home/pokanovic/gdu/results/"
#method="cosine_similarity"
test_source="mnist"
method="projected"
lambda_sparse=0.001
for lambda_orth in 0 0.1 0.01 0.001
do
  for lambda_OLS in 0 0.1 0.01 0.001
  do
      bsub -W 20:00 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" "python SimulationExperiments/experiment_4_digits/digits_5_classification.py --run_all 0 --res_file_dir $file  --method $method --TARGET_DOMAIN $test_source --fine_tune False --lambda_orth ${lambda_orth} --lambda_sparse $lambda_sparse --lambda_OLS $lambda_OLS >& one_heatmap_ORTH_${test_source}_${method}_e2e_${lambda_orth}_${lambda_OLS}.out"
  done
done

# -o output bsub -e error after