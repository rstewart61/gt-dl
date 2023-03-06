import sys
from datasets import load_from_disk

train_dataset = load_from_disk(sys.argv[1])
print('number of samples', train_dataset.num_rows)

