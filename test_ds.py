from datasets import load_dataset
dataset = load_dataset('mlx-community/wikisql', split='train', streaming=True)
dataset_head = dataset.take(10)
print(list(dataset_head))