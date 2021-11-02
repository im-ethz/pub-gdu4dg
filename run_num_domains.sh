file="/cluster/home/pokanovic/gdu/results/"
for num_domains in 2 3 4 5 6 7 8 9 10
do
      bsub -W 20:00 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" "python SimulationExperiments/experiment_4_digits/digits_5_classification.py --run_all 0 --res_file_dir $file  --num_domains $num_domains >& num_domains_${num_domains}.out"
done

# -o output bsub -e error after