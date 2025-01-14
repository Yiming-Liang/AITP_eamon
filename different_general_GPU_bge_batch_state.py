import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from FlagEmbedding import BGEM3FlagModel
from datasets import load_dataset
import random
import time
from datetime import datetime, timedelta
import argparse
import os
import json
import torch
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq



parser = argparse.ArgumentParser(description='Run inference and save results.')
parser.add_argument('--sft_dataset', type=str, help='dataset1 name to use')
parser.add_argument('--sft_dataset_brief', type=str, help='dataset name to use')
parser.add_argument('--sft_nest_path', nargs='+', default=[], help='sft data key to use')
parser.add_argument('--pretrain_dataset', type=str, help='dataset2 name to use')
parser.add_argument('--pretrain_dataset_brief', type=str, help='dataset name to use')
parser.add_argument('--pretrain_nest_path', nargs='+', default=[], help='sft data key to use')
parser.add_argument('--split', default='train', type=str, help='Data split to use')
args = parser.parse_args()
print(args)

function_registry = {}

def register_function(func):
    function_registry[func.__name__] = func
    return func

@register_function
def get_nested_value(item, path):
    fields = path.split('.')
    for field in fields:
        if '[' in field and ']' in field:
            key, index = field[:-1].split('[')
            item = item[key][int(index)]
        else:
            item = item[field]
    return item

@register_function
def filter_for_NoFilter(item):
    return True

@register_function
def filter_for_WildChat(item):
    language = get_nested_value(item, 'conversation[0].language')
    if language != 'English':
        return False
    return True

def filter_all(fucname, item):
    if fucname in function_registry:
        return function_registry[fucname](item)
    else:
        return function_registry['filter_for_NoFilter'](item)

def gpu_standardize_columns_distributed(embeddings, num_gpus=8):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings)

    if not embeddings.is_cuda:
        embeddings = embeddings.cuda()

    num_rows, num_columns = embeddings.shape


    split_size = num_columns // num_gpus  
    splits = [split_size] * (num_gpus - 1) + [num_columns - split_size * (num_gpus - 1)]  


    split_embeddings = torch.split(embeddings, split_size_or_sections=splits, dim=1)

    assert sum([split.shape[1] for split in split_embeddings]) == num_columns, "Column split error!"


    col_means_list = []
    col_stds_list = []
    for i, split in enumerate(split_embeddings):

        split = split.to(f'cuda:{i}')
        col_means = torch.mean(split, dim=0)  
        col_stds = torch.std(split, dim=0)    

        col_means_list.append(col_means.cpu())
        col_stds_list.append(col_stds.cpu())


    col_means = torch.cat(col_means_list, dim=0)
    col_stds = torch.cat(col_stds_list, dim=0)

    print("Column sizes after split:", [split.shape for split in split_embeddings])
    print("column size:", col_means.shape, col_stds.shape)


    epsilon = 1e-8
    col_stds[col_stds == 0] = epsilon


    standardized_splits = []
    for i, split in enumerate(split_embeddings):
        split = split.to(f'cuda:{i}')  
        split_mean = col_means[splits[i - 1] if i > 0 else 0 : splits[i - 1] + split.shape[1] if i > 0 else split.shape[1]].to(f'cuda:{i}')
        split_std = col_stds[splits[i - 1] if i > 0 else 0 : splits[i - 1] + split.shape[1] if i > 0 else split.shape[1]].to(f'cuda:{i}')
        
        standardized_split = (split - split_mean) / split_std
        standardized_splits.append(standardized_split.cpu())  

    standardized_embeddings = torch.cat(standardized_splits, dim=1)

    return standardized_embeddings.numpy()

# Reservoir sampling
def reservoir_sampling(dataset, sample_size):
    reservoir = []
    for idx, example in enumerate(dataset):
        if idx < sample_size:
            reservoir.append(example)
        else:
            j = random.randint(0, idx)
            if j < sample_size:
                reservoir[j] = example
    return reservoir


class CustomDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

random_indices = random.sample(range(30000), 10000)

start_time = time.time()
print(start_time)
start_time_in_file_name = datetime.fromtimestamp(time.time()).strftime('%m_%d_%H_%M_%S')
print(start_time_in_file_name)


# KDE 
def gaussian_kde_gpu(data, points, bandwidth=1.0):
    diff = data[:, None, :] - points[None, :, :]
    dist_sq = torch.sum(diff ** 2, dim=-1)
    density = torch.exp(-0.5 * dist_sq / bandwidth ** 2).mean(dim=0)
    return density
    

def uni_load_data(dataset_path, mode='parquet'):
    if os.path.isdir(dataset_path):
        dataset = load_dataset(dataset_path)
        dataset = dataset[args.split]
    elif os.path.isfile(dataset_path):
        file_extension = os.path.splitext(dataset_path)[1]
        
        if file_extension == '.parquet':
            dataset = load_dataset('parquet', data_files=dataset_path)
            dataset = dataset[args.split]
        elif file_extension == '.jsonl':
            dataset = [json.loads(line.strip()) for line in open(dataset_path, 'r', encoding='utf-8')]
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    else:
        raise ValueError(f"{dataset_path} is not a valid directory or file.")
    
    return dataset





if __name__ == '__main__':
    openhermes_data = uni_load_data(args.sft_dataset)
    cosmopedia_data = uni_load_data(args.pretrain_dataset)

    openhermes_conversations = ['\n'.join([get_nested_value(item, path).strip() for path in args.sft_nest_path]) for item in openhermes_data if filter_all(f"filter_for_{args.sft_dataset_brief}", item)]
    print('openhermes_conversations nums:', len(openhermes_conversations))

    print("Show Processed Case")
    print(openhermes_conversations[0])
    cosmopedia_prompts_texts = ['\n'.join([get_nested_value(item, path).strip() for path in args.pretrain_nest_path]) for item in cosmopedia_data if filter_all(f"filter_for_{args.pretrain_dataset_brief}", item)]

    model = SentenceTransformer("/gpfs/public/research/yiming/eamon/plot_data/models/bge_m3", model_kwargs={"torch_dtype": torch.float16})

    pool = model.start_multi_process_pool()

    batch_size = 1024
    output_dim = 1024
    
    pretrain_state_file_path = f'/gpfs/public/research/yiming/eamon/plot_data/state_data/{args.pretrain_dataset_brief}.npz'
    if os.path.exists(pretrain_state_file_path):
        print(f"File already exists at {pretrain_state_file_path},  so we can load the state vectors of pre-trained data directly \
              to reduce the amount of computation. ")
        pretrain_state_data = np.load(pretrain_state_file_path)
        cosmopedia_embeddings = pretrain_state_data['array1']
        cosmopedia_embeddings_standardized = pretrain_state_data['array2']
    else:
        cosmopedia_dataset = CustomDataset(cosmopedia_prompts_texts)
        cosmopedia_data_loader = DataLoader(
            cosmopedia_dataset,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False,
        )

        cosmopedia_embeddings = np.empty((0, output_dim))
        for batch in tqdm(cosmopedia_data_loader, desc="encoding cosmopedia_embeddings parallelly"):
            with torch.no_grad():
                batch_embeddings = model.encode_multi_process(batch, pool, batch_size=batch_size)
                cosmopedia_embeddings = np.vstack((cosmopedia_embeddings, batch_embeddings))

        print("standardizing cosmopedia embeddings...")
        # openhermes_embeddings_standardized = gpu_standardize_columns(openhermes_embeddings) 
        cosmopedia_embeddings_standardized = gpu_standardize_columns_distributed(cosmopedia_embeddings)
        
        # 保存多个数组到一个文件
        np.savez(pretrain_state_file_path, array1=cosmopedia_embeddings, array2=cosmopedia_embeddings_standardized)
        print(f"File saved to {pretrain_state_file_path}")



    print("performing PCA on cosmopedia embeddings...")
    pca = PCA(n_components=2)
    cosmopedia_pca_result = pca.fit_transform(cosmopedia_embeddings_standardized)
    print("Done 1 !")


    openhermes_dataset = CustomDataset(openhermes_conversations)
    openhermes_data_loader = DataLoader(
        openhermes_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )

    openhermes_embeddings = np.empty((0, output_dim))
    for batch in tqdm(openhermes_data_loader, desc="encoding openhermes_embeddings parallelly"):
        with torch.no_grad():
            batch_embeddings = model.encode_multi_process(batch, pool, batch_size=batch_size)
            openhermes_embeddings = np.vstack((openhermes_embeddings, batch_embeddings))

    model.stop_multi_process_pool(pool)
    
    openhermes_embeddings_standardized = gpu_standardize_columns_distributed(openhermes_embeddings)
    
    openhermes_pca_result = pca.transform(openhermes_embeddings_standardized)
    
    
    

    # np.savez('/gpfs/public/research/gezhang/SHEEP/eamon/plot_data/state_data/Neo_first_phrase.npz', array1=openhermes_embeddings, array2=openhermes_embeddings_standardized)
    # print(f"File saved to Neo_first_phrase.npz")

    cosmopedia_pca_result = torch.tensor(cosmopedia_pca_result, device='cuda:7')  
    openhermes_pca_result = torch.tensor(openhermes_pca_result, device='cuda:7')
    
    # save density results
    all_openhermes_density = []
    all_cosmopedia_density= []

    print("openhermes_pca_result shape:", openhermes_pca_result.shape)
    num_points_openhermes = openhermes_pca_result.shape[0] 
    num_points_cosmopedia = cosmopedia_pca_result.shape[0] 
    batch_size = 128

    caculate_time = time.time()
    for start in range(0, num_points_cosmopedia, batch_size):
        end = min(start + batch_size, num_points_cosmopedia)
        batch_cosmopedia = cosmopedia_pca_result[start:end, :]
        

        openhermes_density_batch = gaussian_kde_gpu(openhermes_pca_result, batch_cosmopedia, bandwidth=1.0)
        openhermes_density_batch = torch.maximum(openhermes_density_batch, torch.tensor(1e-10, device='cuda:7'))  

        cosmopedia_density_batch = gaussian_kde_gpu(cosmopedia_pca_result, batch_cosmopedia, bandwidth=1.0)
        # openhermes_density_batch = torch.maximum(openhermes_density_batch, torch.tensor(1e-10, device='cuda:6'))  
        all_openhermes_density.append(openhermes_density_batch.cpu().numpy()) 
        all_cosmopedia_density.append(cosmopedia_density_batch.cpu().numpy())  

    print("Calculating OpenHermes KDE time:", str(time.time() - caculate_time))
    
    

    openhermes_density = np.concatenate(all_openhermes_density, axis=0)
    cosmopedia_density = np.concatenate(all_cosmopedia_density, axis=0)

    normalized_cosmopedia_density = (cosmopedia_density - np.mean(cosmopedia_density)) / np.std(cosmopedia_density)
    normalized_openhermes_density = (openhermes_density - np.mean(openhermes_density)) / np.std(openhermes_density)
    print(f"cosmopedia_density[:10]:{cosmopedia_density[:10]}")
    print(f"openhermes_density[:10]:{openhermes_density[:10]}")
    ratio = cosmopedia_density/openhermes_density
    # ratio = normalized_cosmopedia_density / normalized_openhermes_density
    

    # # ratio = normalized_cosmopedia_density / normalized_openhermes_density
    # ratio = normalized_openhermes_density
    
    print(f"Greater than 0: {sum(x > 0 for x in ratio)}, Less than 0: {sum(x < 0 for x in ratio)}")
    print(f"Greater than 0.2: {sum(x > 0.2 for x in ratio)}, Less than 0.2: {sum(x < 0.2 for x in ratio)}")
    print(f"Greater than 0.5: {sum(x > 0.5 for x in ratio)}, Less than 0.5: {sum(x < 0.5 for x in ratio)}")
    print(f"Greater than 0.7: {sum(x > 0.7 for x in ratio)}, Less than 0.7: {sum(x < 0.7 for x in ratio)}")
    print(f"Greater than 0.8: {sum(x > 0.8 for x in ratio)}, Less than 0.8: {sum(x < 0.8 for x in ratio)}")
    print(f"Greater than 1: {sum(x > 1 for x in ratio)}, Less than 1: {sum(x < 1 for x in ratio)}")
    print(f"Greater than 2: {sum(x > 2 for x in ratio)}, Less than 2: {sum(x < 2 for x in ratio)}")

    for threshold in [0.2, 0.5, 0.7, 0.8, 1]:
        parquet_data=[]
        for index, point in enumerate(tqdm(cosmopedia_pca_result, desc='save difference...')):
            if ratio[index] < threshold:
                original_data = cosmopedia_data[index]
                parquet_data.append(original_data)
        
        print(f"origin_data_num: {len(parquet_data)}")        
        sample_size = 732000
        sample_data = reservoir_sampling(parquet_data, sample_size)
        
        print(f"save_sample_data_num: {len(sample_data)}")  

        df = pd.DataFrame(sample_data)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, f'../remove_datasets/different_bge_{args.sft_dataset_brief}_{args.pretrain_dataset_brief}_{start_time_in_file_name}_{threshold}.parquet')

    # plt.figure(figsize=(8, 6))
    # sns.kdeplot(x=cosmopedia_pca_result[:, 0], y=cosmopedia_pca_result[:, 1], cmap='Reds', fill=True, label=args.pretrain_dataset_brief, alpha=1)
    # sns.kdeplot(x=openhermes_pca_result[:, 0], y=openhermes_pca_result[:, 1], cmap='Blues', fill=True, label=args.sft_dataset_brief, alpha=0.5)
    # plt.title(f'KDE Plot of {args.pretrain_dataset_brief} and {args.sft_dataset_brief} Datasets')
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')

    # plt.savefig(f'./pictures/bge_{args.sft_dataset_brief}_{args.pretrain_dataset_brief}_kde_{start_time_in_file_name}.pdf')
    # plt.close()

    print(f"end_time:{time.time()}")
    interval_time = time.time() - start_time
    formatted_interval_time = str(timedelta(seconds=interval_time))
    print(f"interval_time: {formatted_interval_time}")

    print("Done.")


