for method in "cosine_similarity" "mmd" "projected"
do
  for fine_tune in "True" "False"
  do
    for running in 0 1 2 3 4
    do
          bsub -W 23:59 -sp 60 -R "rusage[mem=80192]" -R "rusage[scratch=80000,ngpus_excl_p=1]" "python SimulationExperiments/experiments_wilds/wilds_playground.py   --method $method  --fine_tune ${fine_tune}  --running $running >& camelyonrun_${method}_${fine_tune}_${running}.out"
    done
  done
done