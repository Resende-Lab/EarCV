#!/bin/bash
#SBATCH --job-name=9.tmp/3.jobs/array.%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=4GB
#SBATCH --qos=mresende
#SBATCH --account=mresende
#SBATCH -t 1:00:00
#SBATCH --output=9.tmp/3.jobs/%a.array.%A.out
#SBATCH --array=1-1

module load ufrc
printf "experiment\ttrait\tcv\tnrounds\teta\tsubsample\tlambda\talpha\ty\tyhat\tV1\tV2\tV3\n" >> 'results.txt'

for experiment in {1..10}
  do
  	for trait in {2..2}
  	do
  		for cv in {1..10}
  		do
			for nrounds in {500..750..250}
			do
				for eta in $(seq 0.1 0.1 0.2)
				do
					for subsample in 0.8
					do
						for lambda in $(seq 0.2 0.2 0.4)
						do
							for alpha in $(seq 0.6 0.2 0.8)
							do
				  			#	echo $experiment $trait $cv $nrounds $eta $subsample
  								sbatch 2.krn.sh $experiment $trait $cv $nrounds $eta $subsample $lambda $alpha
							done
						done
					done
				done
			done
  		done
  	done
done


