# source /xpfs/public/gezhang/zk/miniconda3/bin/activate shift
cd /gpfs/public/research/yiming/eamon/plot_data/shift_distribution
# dataset1 is sft data, and dataset2 is pretrain data

python -u difference_general_GPU_bge.py \
--sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/tulu-v2-sft-mixture \
--sft_dataset_brief TuluV2 \
--sft_nest_path messages[0].content messages[1].content \
--pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
--pretrain_dataset_brief dolma2532000 \
--pretrain_nest_path text 2>&1 | tee ./logs/difference_bge_tuluv2_dolma_2532000_difference_general_GPU_threshold_1_11_25_1119.log

# python -u smae_distribution_tulu_general_GPU.py \
# --sft_dataset /gpfs/public/research/gezhang/SHEEP/eamon/plot_data/datasets/tulu-v2-sft-mixture \
# --sft_dataset_brief TuluV2 \
# --sft_nest_path messages[0].content messages[1].content \
# --pretrain_dataset /gpfs/public/research/gezhang/SHEEP/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief dolma2532000 \
# --pretrain_nest_path text 2>&1 | tee ./logs/difference_tuluv2_dolma_2532000_difference_general_GPU_threshold_1.log


# python -u difference_general_GPU.py \
# --sft_dataset /gpfs/public/research/gezhang/SHEEP/eamon/plot_data/datasets/tulu-v2-sft-mixture \
# --sft_dataset_brief TuluV2 \
# --sft_nest_path messages[0].content messages[1].content \
# --pretrain_dataset /gpfs/public/research/gezhang/SHEEP/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief dolma2532000 \
# --pretrain_nest_path text 2>&1 | tee ./logs/difference_tuluv2_dolma_2532000_difference_general_GPU_threshold_1.log


# python -u difference_general_GPU.py \
# --sft_dataset /gpfs/public/research/gezhang/SHEEP/eamon/plot_data/datasets/tulu-v2-sft-mixture \
# --sft_dataset_brief TuluV2 \
# --sft_nest_path messages[0].content messages[1].content \
# --pretrain_dataset /gpfs/public/research/gezhang/SHEEP/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief dolma2532000 \
# --pretrain_nest_path text 2>&1 | tee ./logs/difference_tuluv2_dolma_2532000_difference_general_GPU_threshold_0.5.log