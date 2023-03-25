
mkdir -p $SCRATCH/ecoroar/results/masking
lfs find $SCRATCH/ecoroar/results -name "masking_*" -type f | xargs mv -t $SCRATCH/ecoroar/results/masking

mkdir -p $SCRATCH/ecoroar/results/faithfulness
lfs find $SCRATCH/ecoroar/results -name "faithfulness_*" -type f | xargs mv -t $SCRATCH/ecoroar/results/faithfulness
mv  $SCRATCH/ecoroar/intermediate/masked_dataset $SCRATCH/ecoroar/intermediate/faithfulness

mkdir -p $SCRATCH/ecoroar/results/ood
lfs find $SCRATCH/ecoroar/results -name "ood_*" -type f | xargs mv -t $SCRATCH/ecoroar/results/ood
mv  $SCRATCH/ecoroar/intermediate/ood_annotated $SCRATCH/ecoroar/intermediate/ood
