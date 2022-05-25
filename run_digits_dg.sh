timestamp=`date +"%Y%m%d%H%M%S"`
file="/cluster/project/jbuhmann/alinadu/pub-gdu4dg/results/${timestamp}/"
mkdir -p ${file}
mkdir ${timestamp}
lambda_sparse=0.001
lambda_OLS=0.001
lambda_orth=0

for test_source in "mnistm" "syn" "svhn" "mnist"
do
  for method in "None"
  do
    for fine_tune in "True" 
    do
      for running in 0 1 2 3 4 5 6 7 8 9
      do
            bsub -W 23:59 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" -o ${timestamp}/digitsdg_${test_source}_${method}_${fine_tune}_${running}.out -e ${timestamp}/digitsdg_${test_source}_${method}_${fine_tune}_${running}.err "python SimulationExperiments/digits5/digits_dg_classification.py --run_all 0 --res_file_dir ${file}  --method ${method} --TARGET_DOMAIN ${test_source} --fine_tune ${fine_tune} --lambda_orth ${lambda_orth} --lambda_sparse ${lambda_sparse} --lambda_OLS ${lambda_OLS} --running ${running}"
      done
    done
  done
done


#for test_source in "mnistm" "syn" "svhn" "mnist"
#do
#  for method in "cosine_similarity" "MMD" "projected"
#  do
#    for fine_tune in "False"
#    do
#      for running in 0 1 2 3 4 5 6 7 8 9
#      do
#            bsub -W 23:59 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" -o ${timestamp}/digitsdg_${test_source}_${method}_${fine_tune}_${running}.out -e ${timestamp}/digitsdg_${test_source}_${method}_${fine_tune}_${running}.err "python SimulationExperiments/digits5/digits_dg_classification.py --run_all 0 --res_file_dir ${file}  --method ${method} --TARGET_DOMAIN ${test_source} --fine_tune ${fine_tune} --lambda_orth ${lambda_orth} --lambda_sparse ${lambda_sparse} --lambda_OLS ${lambda_OLS} --running ${running}"
#      done
#    done
#  done
#done
