#!/bin/bash
echo ${BASH_VERSION}

timestamp=`date +"%Y%m%d%H%M%S"`

#timestamp="20210819094725"
mkdir -p "/cluster/home/alinadu/GDU/pub-gdu4dg/results/ensemble/${timestamp}"
for test_source in "mnistm" "mnist" "syn" "svhn" "usps"
do
	folder_name="${timestamp}/ensemble"
	mkdir -p ${folder_name}
	filename="${folder_name}/target_${test_source}"
	for i in 0 1 2 3 4 
	do
        	bsub -W 3:59 -R "rusage[mem=80192]" -R "rusage[scratch=1000,ngpus_excl_p=1]" -o ${filename}.out -e ${filename}.err python digits_5_classification_ensemble.py  --target=${test_source} --timestamp=${timestamp} --filename="target_${test_source}_ensemble" --run_id=${i}
	done
done


