for method in  "None"
do
  for fine_tune in "True"
  do
    for running in 0 1 2
    do
          bsub -W 48:00 -sp 80 -R "rusage[mem=180192]" -R "rusage[scratch=130000,ngpus_excl_p=1]" "python SimulationExperiments/experiments_wilds/rxrx1/rxrx1_main.py   --method $method  --fine_tune ${fine_tune}  --running $running >&  wildcam_new_runs_${method}_${fine_tune}_${running}.out"
    done
  done
done
