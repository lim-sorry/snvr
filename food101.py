import os

with open('../data/food101/meta/train.txt', 'r') as f:
    train_lines = f.readlines()
    for line in train_lines:
        line = line.strip()
        os.makedirs(os.path.join('../data/food101/train', line.split('/')[0]), exist_ok=True)
        os.rename(os.path.join('../data/food101/images', line+'.jpg'), os.path.join('../data/food101/train', line+'.jpg'))
        print(f"Moved {line} to train set")

with open('../data/food101/meta/test.txt', 'r') as f:
    test_lines = f.readlines()
    for line in test_lines:
        line = line.strip()
        os.makedirs(os.path.join('../data/food101/test', line.split('/')[0]), exist_ok=True)
        os.rename(os.path.join('../data/food101/images', line+'.jpg'), os.path.join('../data/food101/test', line+'.jpg'))
        print(f"Moved {line} to test set")