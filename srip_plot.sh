file="/cluster/home/pokanovic/gdu/results/"
#method="cosine_similarity"
#test_source="mnist"
method="projected"
lambda_sparse=0.001
lambda_OLS=0.001
for test_source in "mnistm" "syn" "svhn" "usps"
do
  for lambda_orth in 0 0.01
  do
    for fine_tune in "True" "False"
    do
      for running in 0 1 2 3 4
      do
            bsub -W 20:00 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" "python SimulationExperiments/experiment_4_digits/digits_5_classification.py --run_all 0 --res_file_dir $file  --method $method --TARGET_DOMAIN $test_source --fine_tune ${fine_tune} --lambda_orth ${lambda_orth} --lambda_sparse $lambda_sparse --lambda_OLS $lambda_OLS --running $running >& initsrip__${test_source}_${method}_${fine_tune}_${lambda_orth}_${running}.out"
      done
    done
  done
done

# -o output bsub -e error after