file="/cluster/home/pokanovic/gdu/results/"
for method in "cosine_similarity" "MMD" "projected"
do
  for test_source in "mnist" "mnistm" "syn" "svhn" "usps"
  do
    for fine_tune in "True" "False"
    do
      bsub -W 20:00 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" "python SimulationExperiments/experiment_4_digits/digits_5_classification.py --run_all 0 --res_file_dir $file  --method $method --TARGET_DOMAIN $test_source --fine_tune $fine_tune --lambda_orth 0 --lambda_sparse 0.001 --lambda_OLS 0.001 >& frozen${test_source}_${method}_${fine_tune}.out"
    done
  done
done

