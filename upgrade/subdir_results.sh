
mkdir -p ~/scratch/ecoroar/results/masking
lfs find ~/scratch/ecoroar/results  -maxdepth 1 -name "masking_*" -type f | xargs mv -v -t ~/scratch/ecoroar/results/masking

mkdir -p ~/scratch/ecoroar/results/faithfulness
lfs find ~/scratch/ecoroar/results  -maxdepth 1 -name "faithfulness_*" -type f | xargs mv -v -t ~/scratch/ecoroar/results/faithfulness
mv  ~/scratch/ecoroar/intermediate/masked_dataset ~/scratch/ecoroar/intermediate/faithfulness

mkdir -p ~/scratch/ecoroar/results/ood
lfs find ~/scratch/ecoroar/results  -maxdepth 1 -name "ood_*" -type f | xargs mv -v -t ~/scratch/ecoroar/results/ood
mv  ~/scratch/ecoroar/intermediate/ood_annotated ~/scratch/ecoroar/intermediate/ood
