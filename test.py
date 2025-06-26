import os

path_list = os.listdir('../data/kma_data/nc')
path_list = sorted(path_list)

for i, path in enumerate(path_list):
    file_list = os.listdir(os.path.join('../data/kma_data/nc', path))
    print(f'path: {path}, num files: {len(file_list)}') if i % 10 == 0 else None