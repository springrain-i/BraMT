import scipy
from scipy import signal
import os
import lmdb
import pickle
import numpy as np
import glob

labels = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8])
root_dir = '../Raw_data/FACED'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)

files_dict = {
    'train':files[:80],
    'val':files[80:100],
    'test':files[100:],
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}
os.makedirs('../data/Faced/processed', exist_ok=True)
db = lmdb.open('../data/Faced/processed', map_size=6612500172)

for files_key in files_dict.keys():
    for dir in files_dict[files_key]:
        # 现在的file是subxxx
        mat = glob.glob(os.path.join(root_dir, dir, '*.mat'))[0]
        print(mat)
        mat = scipy.io.loadmat(mat)
        print(mat.keys())  # 查看有哪些键
        print(mat['After_remark'].shape)
        print(mat['After_remark'])
        #array = pickle.load(f)
        #print(file)
        break         
    break
        eeg = signal.resample(array, 6000, axis=2)
        eeg_ = eeg.reshape(28, 32, 30, 200)
        for i, (samples, label) in enumerate(zip(eeg_, labels)):
            for j in range(3):
                sample = samples[:, 10*j:10*(j+1), :]
                sample_key = f'{file}-{i}-{j}'
                print(sample_key)
                data_dict = {
                    'sample': sample, 'label': label
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()