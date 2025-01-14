
source /gpfs/public/research/miniconda3/bin/activate shift
cd /gpfs/public/research/yiming/eamon/plot_data/shift_distribution
set -e 
set -x
pip install sentence_transformers
# dataset1 is sft data, and dataset2 is pretrain data

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u data_tsne_ddp.py 2>&1 | tee ./logs/tsne_100_500_black_on.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state_coordinate_select.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/shift_distribution/data_with_coordinate.jsonl \
# --sft_dataset_brief Rewrite \
# --sft_nest_path query answer \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/look_coordinate_remove_bge_tulu_all_extract_qa.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state_coordinate_select.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/shift_distribution/data_with_coordinate.jsonl \
# --sft_dataset_brief Rewrite \
# --sft_nest_path query answer \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/look_coordinate_remove_bge_tulu_all_extract_qa.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state_coordinate_select.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/tulu-v2-sft-mixture \
# --sft_dataset_brief TuluV2 \
# --sft_nest_path messages[0].content messages[1].content \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/look_tulu_coordinate_remove_bge_tulu_all_extract_qa.log



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state_coordinate_select.py \
# --sft_dataset /gpfs/public/research/tianyu/shift/remove_bge_tulu/all_extract_qa.jsonl \
# --sft_dataset_brief Rewrite \
# --sft_nest_path text \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/look_coordinate_remove_bge_tulu_all_extract_qa.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/tulu-v2-sft-mixture \
# --sft_dataset_brief TuluV2 \
# --sft_nest_path messages[0].content messages[1].content \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/TuluV2_dolma_2532000_plot_general_bge.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/OpenHermes_2 \
# --sft_dataset_brief OpenHermes \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_cosmopedia_1553237_0.05.parquet \
# --pretrain_dataset_brief Cosmopedia \
# --pretrain_nest_path text 2>&1 | tee ./logs/Cosmopedia_1553237_OpenHermes_plot_general_bge_state.log



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/remove_datasets/remove_bge_TuluV2_dolma2532000_10_08_09_01_50_0.7.parquet \
# --sft_dataset_brief Remove_Tulu \
# --sft_nest_path text \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_remove_Tulu_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/tianyu/shift/remove_bge_tulu/all_extract_qa.jsonl \
# --sft_dataset_brief Remove_Rewrite \
# --sft_nest_path query answer \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_rewrite_remove_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/merged_bge_remove_07_and_tulu_v2_sft_mixture.jsonl \
# --sft_dataset_brief Remove_Merge \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_merge_remove_rewrite_and_Tulu_plot_general_bge_state.log



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/remove_datasets/different_bge_TuluV2_Dolma_11_26_12_11_15_1.parquet \
# --sft_dataset_brief Different_Tulu \
# --sft_nest_path text \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_Different_Tulu_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/tianyu/shift/different_bge_TuluV2_Dolma/all_extract_qa_filtered.jsonl \
# --sft_dataset_brief Different_Rewrite \
# --sft_nest_path query answer \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_rewrite_different_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/merged_bge_different_rewrite_and_tulu.jsonl \
# --sft_dataset_brief Different_Merge \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_merge_different_rewrite_and_Tulu_plot_general_bge_state.log



# ################################################ different 005 projection into dolma ####################################################
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/ratio/sampled_different_bge_dolma_rewrite_0.05_ratio.jsonl \
# --sft_dataset_brief Different_Tulu_005 \
# --sft_nest_path text[0] \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_different_Tulu_005_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/ratio/sampled_different_bge_dolma_rewrite_0.05_ratio.jsonl \
# --sft_dataset_brief Different_Rewrite_005 \
# --sft_nest_path query answer \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_different_rewrite_005_remove_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/ratio/merged_different_bge_dolma_rewrite_0.05_ratio_and_tulu.jsonl \
# --sft_dataset_brief Merge_Different_005 \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_merge_005_different_rewrite_and_Tulu_plot_general_bge_state.log


# ################################################ remove 005 projection into dolma ####################################################
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/ratio/sampled_bge_dolma_rewrite_0.05_ratio.jsonl \
# --sft_dataset_brief Remove_Tulu_005 \
# --sft_nest_path text[0] \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_remove_Tulu_005_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/ratio/sampled_bge_dolma_rewrite_0.05_ratio.jsonl \
# --sft_dataset_brief Remove_Rewrite_005 \
# --sft_nest_path query answer \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_remove_rewrite_005_remove_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/ratio/merged_bge_dolma_rewrite_0.05_ratio_and_tulu.jsonl \
# --sft_dataset_brief Merge_Remove_005 \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_merge_005_remove_rewrite_and_Tulu_plot_general_bge_state.log



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/remove_datasets/remove_bge_TuluV2_dolma2532000_10_08_09_01_50_0.7.parquet \
# --sft_dataset_brief Remove_Tulu \
# --sft_nest_path text \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_remove_Tulu_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/tianyu/shift/remove_bge_tulu/all_extract_qa.jsonl \
# --sft_dataset_brief Rewrite \
# --sft_nest_path query answer \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_rewrite_remove_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/merged_bge_remove_07_and_tulu_v2_sft_mixture.jsonl \
# --sft_dataset_brief Merge \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_merge_remove_rewrite_and_Tulu_plot_general_bge_state.log






# ############################### SFT datasets project into Dolma pretrain datasets ################################ 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/WildChat \
# --sft_dataset_brief WildChat \
# --sft_nest_path conversation[0].content conversation[1].content \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_WildChat_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/WizardLM_evol_instruct_V2_196k \
# --sft_dataset_brief EvolInstruct \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_EvolInstruct_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/databricks-dolly-15k \
# --sft_dataset_brief Dolly \
# --sft_nest_path instruction context response \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_Dolly_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/tulu-v2-sft-mixture \
# --sft_dataset_brief TuluV2 \
# --sft_nest_path messages[0].content messages[1].content \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_TuluV2_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/OpenHermes_2 \
# --sft_dataset_brief OpenHermes \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_OpenHermes_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/filtered_neo_two_phrase/merged_neo_first_phrase.jsonl \
# --sft_dataset_brief Neo_SFT \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_Neo_SFT_plot_general_bge_state.log









# ############################### SFT datasets project into Cosmopeida pretrain datasets ################################ 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/WildChat \
# --sft_dataset_brief WildChat \
# --sft_nest_path conversation[0].content conversation[1].content \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_cosmopedia_1553237_0.05.parquet \
# --pretrain_dataset_brief Cosmopedia \
# --pretrain_nest_path text 2>&1 | tee ./logs/Cosmopedia_1553237_WildChat_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/WizardLM_evol_instruct_V2_196k \
# --sft_dataset_brief EvolInstruct \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_cosmopedia_1553237_0.05.parquet \
# --pretrain_dataset_brief Cosmopedia \
# --pretrain_nest_path text 2>&1 | tee ./logs/Cosmopedia_1553237_EvolInstruct_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/databricks-dolly-15k \
# --sft_dataset_brief Dolly \
# --sft_nest_path instruction context response \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_cosmopedia_1553237_0.05.parquet \
# --pretrain_dataset_brief Cosmopedia \
# --pretrain_nest_path text 2>&1 | tee ./logs/Cosmopedia_1553237_Dolly_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/tulu-v2-sft-mixture \
# --sft_dataset_brief TuluV2 \
# --sft_nest_path messages[0].content messages[1].content \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_cosmopedia_1553237_0.05.parquet \
# --pretrain_dataset_brief Cosmopedia \
# --pretrain_nest_path text 2>&1 | tee ./logs/Cosmopedia_1553237_TuluV2_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/OpenHermes_2 \
# --sft_dataset_brief OpenHermes \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_cosmopedia_1553237_0.05.parquet \
# --pretrain_dataset_brief Cosmopedia \
# --pretrain_nest_path text 2>&1 | tee ./logs/Cosmopedia_1553237_OpenHermes_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/filtered_neo_two_phrase/merged_neo_first_phrase.jsonl \
# --sft_dataset_brief Neo_SFT \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_cosmopedia_1553237_0.05.parquet \
# --pretrain_dataset_brief Cosmopedia \
# --pretrain_nest_path text 2>&1 | tee ./logs/Cosmopedia_1553237_Neo_SFT_plot_general_bge_state.log

# ################################ SFT datasets project into Matrix pretrain datasets ################################ 

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/WildChat \
# --sft_dataset_brief WildChat \
# --sft_nest_path conversation[0].content conversation[1].content \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_matrix_2310504_000050.parquet \
# --pretrain_dataset_brief Matrix \
# --pretrain_nest_path text 2>&1 | tee ./logs/Matrix_2310504_WildChat_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/WizardLM_evol_instruct_V2_196k \
# --sft_dataset_brief EvolInstruct \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_matrix_2310504_000050.parquet \
# --pretrain_dataset_brief Matrix \
# --pretrain_nest_path text 2>&1 | tee ./logs/Matrix_2310504_EvolInstruct_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/databricks-dolly-15k \
# --sft_dataset_brief Dolly \
# --sft_nest_path instruction context response \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_matrix_2310504_000050.parquet \
# --pretrain_dataset_brief Matrix \
# --pretrain_nest_path text 2>&1 | tee ./logs/Matrix_2310504_Dolly_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/tulu-v2-sft-mixture \
# --sft_dataset_brief TuluV2 \
# --sft_nest_path messages[0].content messages[1].content \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_matrix_2310504_000050.parquet \
# --pretrain_dataset_brief Matrix \
# --pretrain_nest_path text 2>&1 | tee ./logs/Matrix_2310504_TuluV2_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/OpenHermes_2 \
# --sft_dataset_brief OpenHermes \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_matrix_2310504_000050.parquet \
# --pretrain_dataset_brief Matrix \
# --pretrain_nest_path text 2>&1 | tee ./logs/Matrix_2310504_OpenHermes_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/filtered_neo_two_phrase/merged_neo_first_phrase.jsonl \
# --sft_dataset_brief Neo_SFT \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_matrix_2310504_000050.parquet \
# --pretrain_dataset_brief Matrix \
# --pretrain_nest_path text 2>&1 | tee ./logs/Matrix_2310504_Neo_SFT_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/remove_datasets/remove_bge_NeoSft_Matrix_10_15_05_11_23_0.7.parquet \
# --sft_dataset_brief Remove_Neo_SFT \
# --sft_nest_path text \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_matrix_2310504_000050.parquet \
# --pretrain_dataset_brief Matrix \
# --pretrain_nest_path text 2>&1 | tee ./logs/Matrix_2310504_remove_Neo_SFT_plot_general_bge_state.log








# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/remove_datasets/remove_bge_TuluV2_dolma2532000_10_08_09_01_50_0.7.parquet \
# --sft_dataset_brief Remove_Tulu \
# --sft_nest_path text \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_remove_Tulu_plot_general_bge_state.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/tianyu/shift/remove_bge_tulu/all_extract_qa.jsonl \
# --sft_dataset_brief Rewrite \
# --sft_nest_path query answer \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_rewrite_remove_plot_general_bge_state.log


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/merged_bge_remove_07_and_tulu_v2_sft_mixture.jsonl \
# --sft_dataset_brief Merge \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/sampled_pretrain_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief Dolma \
# --pretrain_nest_path text 2>&1 | tee ./logs/Dolma_2532000_merge_remove_rewrite_and_Tulu_plot_general_bge_state.log





# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/tulu-v2-sft-mixture \
# --sft_dataset_brief TuluV2 \
# --sft_nest_path messages[0].content messages[1].content \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief dolma_2532000 \
# --pretrain_nest_path text 2>&1 | tee ./logs/TuluV2_dolma_2532000_plot_general_bge.log

# sleep 10

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/merged_remove_07_and_tulu_v2_sft_mixture.jsonl \
# --sft_dataset_brief MergedRemove07 \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief dolma_2532000 \
# --pretrain_nest_path text 2>&1 | tee ./logs/mergedRemove07_dolma_2532000_plot_general_bge.log

# sleep 10


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/remove_TuluV2_dolma2532000_09_26_15_50_41_0.7.parquet \
# --sft_dataset_brief DolmaRemoveTulu07 \
# --sft_nest_path text \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief dolma_2532000 \
# --pretrain_nest_path text 2>&1 | tee ./logs/DolmaRemoveTulu07_dolma_2532000_plot_general_bge.log

# sleep 10

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge_state.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/different_TuluV2_dolma2532000_09_18_08_32_24_1.parquet \
# --sft_dataset_brief DolmaDifferentTulu1 \
# --sft_nest_path text \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief dolma_2532000 \
# --pretrain_nest_path text 2>&1 | tee ./logs/DolmaDifferentTulu1_dolma_2532000_plot_general_bge.log

# sleep 10

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u plot_general_bge.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/merged_difference_1_rewrite_tulu.jsonl \
# --sft_dataset_brief MergedDifferentTulu1 \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief dolma_2532000 \
# --pretrain_nest_path text 2>&1 | tee ./logs/mergedDifferent1_dolma_2532000_plot_general_bge.log



# CUDA_VISIBLE_DEVICES=0 python -u plot_general_bge.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/merged_datasets/merged_difference_1_rewrite_tulu.jsonl \
# --sft_dataset_brief MergedDifferentTulu1 \
# --sft_nest_path conversations[0].value conversations[1].value \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief dolma_2532000 \
# --pretrain_nest_path text 2>&1 | tee ./logs/mergedDifferent1_dolma_2532000_plot_general.log

# sleep 10

# CUDA_VISIBLE_DEVICES=0 python -u plot_general_bge.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/datasets/tulu-v2-sft-mixture \
# --sft_dataset_brief TuluV2 \
# --sft_nest_path messages[0].content messages[1].content \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief dolma_2532000 \
# --pretrain_nest_path text 2>&1 | tee ./logs/TuluV2_dolma_2532000_plot_general.log

# sleep 10

# CUDA_VISIBLE_DEVICES=0 python -u plot_general_bge.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/different_TuluV2_dolma2532000_09_18_08_32_24_1.parquet \
# --sft_dataset_brief DolmaDifferentTulu1 \
# --sft_nest_path text \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief dolma_2532000 \
# --pretrain_nest_path text 2>&1 | tee ./logs/DolmaDifferentTulu1_dolma_2532000_plot_general.log

# sleep 10

# CUDA_VISIBLE_DEVICES=0 python -u plot_general_bge.py \
# --sft_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/remove_TuluV2_dolma2532000_09_26_15_50_41_0.7.parquet \
# --sft_dataset_brief DolmaRemoveTulu07 \
# --sft_nest_path text \
# --pretrain_dataset /gpfs/public/research/yiming/eamon/plot_data/difference_datasets/sample_dolma_2532000_09_15_08_30_56.parquet \
# --pretrain_dataset_brief dolma_2532000 \
# --pretrain_nest_path text 2>&1 | tee ./logs/DolmaRemoveTulu07_dolma_2532000_plot_general.log

