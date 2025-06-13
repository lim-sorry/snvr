import os

main_path = '/home/work/js/repo/data/kma_data/nc/'
path_list = os.listdir(main_path)
for path in path_list:
    file_list =os.listdir(os.path.join(main_path, path))
    for file in file_list:
        if file.endswith('.npy'):
            os.remove(os.path.join(main_path, path, file))
            print(f"Removed file: {os.path.join(main_path, path, file)}")
            