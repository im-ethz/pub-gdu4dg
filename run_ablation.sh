file="/cluster/home/pokanovic/gdu/results/"
for test_source in "mnistm"
do
	for method in "projected"
	do
	  for run_exp in 0 1
	  do
        	bsub -W 20:00 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" "python SimulationExperiments/experiment_4_digits/digits_5_classification.py --run_all 0 --res_file_dir $file  --method $method --TARGET_DOMAIN $test_source --fine_tune True --lambda_orth 0.01 --running $run_exp >& srip_${test_source}_${method}_ft_${run_exp}.out"
	  done
	done
done