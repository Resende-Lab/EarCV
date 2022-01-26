#!/bin/bash
#SBATCH --job-name=9.tmp/3.jobs/array.%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem=4GB
#SBATCH --qos=mresende-b
#SBATCH --account=mresende
#SBATCH -t 0:30:00
#SBATCH --output=9.tmp/3.jobs/%a.array.%A.out
#SBATCH --array=1-1

module load R
module load ufrc

experiment=$1
trait=$2
cv=$3
nrounds=$4
eta=$5
subsample=$6
lambda=$7
alpha=$8

#echo $experiment $trait $cv $nrounds $eta $subsample
Rscript 3.krn.R $experiment $trait $cv $nrounds $eta $subsample $lambda $alpha

