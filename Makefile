
.PHONY: noop download-intermediate-cedar download-table-cedar download-checkpoints-cedar download-tensorboard-cedar download-results-cedar upload-code-cedar upload-code-mila

noop:
	echo "preventing default action"

plot-faithfulness:
	python export/faithfulness_plot.py --model-category size --masking-strategy half-det --split train
	python export/faithfulness_plot.py --model-category size --masking-strategy half-det --split test
	python export/faithfulness_plot.py --model-category size --masking-strategy uni --split test
	python export/faithfulness_plot.py --model-category size --masking-strategy uni --split train

plot-ood:
	python export/ood_plot.py --model-category size --masking-strategy half-det --split train
	python export/ood_plot.py --model-category size --masking-strategy half-det --split test
	python export/ood_plot.py --model-category size --masking-strategy uni --split test
	python export/ood_plot.py --model-category size --masking-strategy uni --split train

plot-train:
	python export/epoch_by_mmr_plot.py --model-category size --masking-strategy uni
	python export/epoch_by_ms_plot.py --model-category size --max-masking-ratio 100
	python export/masked_performance_by_mmr_plot.py --model-category size --masking-strategy uni
	python export/masked_performance_by_ms_plot.py --model-category size --max-masking-ratio 100
	python export/unmasked_performance_by_mmr_plot.py --model-category size --masking-strategy uni
	python export/unmasked_performance_by_ms_plot.py --model-category size --max-masking-ratio 100

plot: plot-train plot-faithfulness plot-ood

appendix-train:
	python export/unmasked_performance_by_valid_ms_plot.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --model-category size --max-masking-ratio 100 --format appendix
	python export/masked_100p_performance_by_valid_ms_plot.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --model-category size --max-masking-ratio 100 --format appendix

appendix-ood:
	python export/ood_plot.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --model roberta-sl --max-masking-ratio 100 --masking-strategy half-det --validation-dataset both --dist-repeat 1 --method simes --threshold 0.05 --format appendix
	python export/ood_plot.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --model  roberta-sb --max-masking-ratio 100 --masking-strategy half-det --validation-dataset both --dist-repeat 1 --method simes --threshold 0.05 --format appendix

appendix-epoch:
	python export/epoch_plot.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --model-category size --max-masking-ratio 100 --masking-strategy half-det --validation-dataset both --format appendix

appendix-faithfulness:
	python export/faithfulness_plot.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --model roberta-sb --max-masking-ratio 100 --masking-strategy half-det --validation-dataset both --format appendix
	python export/faithfulness_plot.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --model roberta-sl --max-masking-ratio 100 --masking-strategy half-det --validation-dataset both --format appendix

appendix-tables:
	python export/datasets_table.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --format appendix
	python export/models_table.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --format appendix
	python export/walltime_fine-tune_table.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --format appendix
	python export/walltime_importance-measure_table.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --format appendix
	python export/walltime_faithfulness_table.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --format appendix
	python export/walltime_ood_table.py --dataset bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --format appendix

appendix: appendix-train appendix-ood appendix-epoch appendix-faithfulness appendix-tables

paper-train:
	python export/unmasked_performance_by_valid_ms_plot.py --dataset MRPC BoolQ --aggregate bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --model-category size --max-masking-ratio 100 --format paper
	python export/masked_100p_performance_by_valid_ms_plot.py --dataset MRPC BoolQ --aggregate bAbI-1 bAbI-2 bAbI-3 BoolQ CB CoLA MIMIC-a MIMIC-d MRPC RTE SST2 --model-category size --max-masking-ratio 100 --format paper

paper-ood:
	python export/ood_plot.py --dataset MRPC BoolQ --model roberta-sl --max-masking-ratio 100 --masking-strategy half-det --validation-dataset both --dist-repeat 1 --method simes --threshold 0.05 --format paper

paper-faithfulness:
	python export/faithfulness_plot.py --dataset MRPC BoolQ --model-category size --max-masking-ratio 100 --masking-strategy half-det --validation-dataset both --format paper

paper: paper-train paper-ood paper-faithfulness

download-intermediate-cedar:
	rsync --info=progress2 -urltv --delete \
		-e ssh cc-cedar:~/scratch/ecoroar/intermediate/ ./intermediate

download-table-cedar:
	rsync --info=progress2 -urltv --delete \
		-e ssh cc-cedar:~/scratch/ecoroar/tables/ ./tables

download-checkpoints-cedar:
	rsync --info=progress2 -urltv --delete \
		-e ssh cc-cedar:~/scratch/ecoroar/checkpoints/masking_m-roberta-sb_d-sst2_s-0_e-10_r-100_y-half-det/ ./checkpoints/masking_m-roberta-sb_d-sst2_s-0_e-10_r-100_y-half-det

download-tensorboard-cedar:
	rsync --info=progress2 -urltv --delete \
		-e ssh cc-cedar:~/scratch/ecoroar/tensorboard/ ./tensorboard

download-results-cedar:
	rsync --info=progress2 -urltvW --delete \
		-e ssh cc-cedar:~/scratch/ecoroar/results/ ./results

download-cache-cedar:
	rsync --info=progress2 -urltv --delete \
		-e ssh cc-cedar:~/scratch/ecoroar/cache/ ./cache

upload-code-cedar:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./ cc-cedar:~/workspace/economical-roar

upload-mimic-cedar:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./mimic/ cc-cedar:~/scratch/ecoroar/mimic

upload-code-mila:
	rsync --info=progress2 -urltv --delete \
		--filter=':- .gitignore' --exclude='.git/' \
		-e ssh ./ mila:~/workspace/economical-roar
