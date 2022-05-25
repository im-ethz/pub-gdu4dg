file="gdu_10_runs/results/"
lambda_sparse=0.001
lambda_OLS=0.001
lambda_orth=0

# End-to-end experiments (E2E)
for test_source in "mnistm" "syn" "svhn" "usps" "mnist"
do
  for method in "cosine_similarity" "MMD" "projected"
  do
    for fine_tune in "False"
    do
      for running in 0 1 2 3 4 5 6 7 8 9
      do
            bsub -W 20:00 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" "python SimulationExperiments/digits5/digits_5_classification.py --run_all 0 --res_file_dir $file  --method $method --TARGET_DOMAIN $test_source --fine_tune ${fine_tune} --lambda_orth ${lambda_orth} --lambda_sparse $lambda_sparse --lambda_OLS $lambda_OLS --running $running >& tenruns_${test_source}_${method}_${fine_tune}_${running}.out"
      done
    done
  done
done

# Fine-tuning experiments (FT) as method should be set to None and fint_tune to "True" _simultaneoulsy_ 
for test_source in "mnistm" "syn" "svhn" "usps" "mnist"
do
  for method in "None"
  do
    for fine_tune in "True"
    do
      for running in 0 1 2 3 4 5 6 7 8 9
      do
            bsub -W 20:00 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" "python SimulationExperiments/digits5/digits_5_classification.py --run_all 0 --res_file_dir $file  --method $method --TARGET_DOMAIN $test_source --fine_tune ${fine_tune} --lambda_orth ${lambda_orth} --lambda_sparse $lambda_sparse --lambda_OLS $lambda_OLS --running $running >& tenruns_${test_source}_${method}_${fine_tune}_${running}.out"
      done
    done
  done
done

# -o output bsub -e error after
