import mne
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
import matplotlib.pyplot as plt
import os
import re
import lmdb
from tqdm import tqdm
import pickle
import numpy as np
import warnings

montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
# silence MNE info/debug so tqdm remains clean
mne.set_log_level('ERROR')  
warnings.filterwarnings("ignore", category=RuntimeWarning)

def _accumulate_duration_seconds(raw_obj: mne.io.Raw, total_seconds: list):
    """Add raw duration (in seconds) into the running accumulator."""
    try:
        n_samples = raw_obj.n_times
    except Exception:
        n_samples = len(raw_obj.times)
    sfreq = raw_obj.info.get('sfreq', None) or 0
    if sfreq:
        total_seconds[0] += n_samples / sfreq

def total_hours(total_seconds: list) -> float:
    """Convert accumulated seconds to hours."""
    return total_seconds[0] / 3600.0

def read_sub_tasks(sub_id, data_path):
    """Reads raw EEG data for a given subject ID."""
    file_path = f"{data_path}/sub-{sub_id}/eeg/sub-{sub_id}_task-"
    tasks = []

    # an example of file name : sub-NDARAB678VYW_task-contrastChangeDetection_run-1_channels.set
    for file in os.listdir(os.path.dirname(file_path)):
        if file.startswith(os.path.basename(file_path)) and file.endswith(".set"):
            match = re.search(r'task-[^_]+' , file)
            if match:
                tasks.append(match.group(0).replace("task-", ""))
    return tasks

def segment_and_store(raw_obj: mne.io.Raw, prefix: str, db: lmdb.Environment, file_key_list: list, rejected_seconds: list):
    """Resample/filter/notch, cut edges, segment to seq_len=30 at 200Hz, and store to LMDB."""
    # TUEG-style processing
    raw_obj.resample(200)
    raw_obj.filter(0.3, 75) # 原本是(1.,100.)
    raw_obj.notch_filter(60)

    data = raw_obj.get_data()  # (chs, times)
    data = data.T  # (times, chs)

    data = data*1e4  # convert from V to 100uV

    #print(data)
    points, chs = data.shape
    if points < 300 * 200:
        rejected_seconds[0] += points / 200.0
        print('Too short, skipping')
        return

    seq_len = 30
    samples_per_seq = seq_len * 200
    a = points % samples_per_seq
    start_cut = 60 * 200
    end_cut = a + 60 * 200
    if end_cut == 0:
        data2 = data[start_cut:, :]
    else:
        data2 = data[start_cut:-end_cut, :]

    data2 = data2.reshape(-1, seq_len, 200, chs)
    data2 = data2.transpose(0, 3, 1, 2)  # (n, chs, seq_len, 200)

    for i, sample in enumerate(data2):
        if np.max(np.abs(sample)) < 100:
            key = f'{prefix}_{i}'
            file_key_list.append(key)
            txn = db.begin(write=True)
            txn.put(key=key.encode(), value=pickle.dumps(sample))
            txn.commit()


def read_sub_data(sub_id, task_name, db: lmdb.Environment, file_key_list: list, total_seconds: list, rejected_seconds: list):
    """Read BIDS raw for a subject/task, segment, store into LMDB, and accumulate duration."""
    print(f"Reading data for subject {sub_id}, task {task_name}")
    bids_path = BIDSPath(subject=sub_id, task=task_name, root=data_path)
    bids_path = BIDSPath(root=data_path, subject=sub_id, task=task_name)
    all_paths = bids_path.match()
    runs = sorted(set(p.run for p in all_paths if p.run is not None))

    if not runs:
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.set_montage(montage)
        raw.set_channel_types({ch: 'eeg' for ch in raw.info['ch_names']})
        raw.drop_channels(['Cz'])
        _accumulate_duration_seconds(raw, total_seconds)
        segment_and_store(raw, f'sub-{sub_id}_task-{task_name}', db, file_key_list, rejected_seconds)
    else:
        for run in runs:
            bids_path_run = bids_path.copy().update(run=run)
            raw_run = read_raw_bids(bids_path=bids_path_run, verbose=False)
            raw_run.set_montage(montage)
            raw_run.set_channel_types({ch: 'eeg' for ch in raw_run.info['ch_names']})
            raw_run.drop_channels(['Cz'])
            _accumulate_duration_seconds(raw_run, total_seconds)
            segment_and_store(raw_run, f'sub-{sub_id}_task-{task_name}_run-{run}', db, file_key_list, rejected_seconds)

def find_sub_id(data_path):
    sub_ids = []
    for entry in os.listdir(data_path):
        if entry.startswith('sub-'):
            sub_ids.append(entry.replace('sub-', ''))
    return sub_ids


if __name__ == "__main__":
    data_path = 'Raw_data/HBN-Release11'
    sub_ids = find_sub_id(data_path)
    #print(sub_ids)
    os.makedirs("data_for_pretrain", exist_ok=True)
    db = lmdb.open("data_for_pretrain/HBN11", map_size=int(1e12))
    file_key_list = []
    total_seconds = [0.0]
    rejected_seconds = [0.0]
    for sub_id in tqdm(sub_ids, desc="Subjects"):
        tasks = read_sub_tasks(sub_id, data_path)
        for task in tqdm(tasks, desc=f"Tasks for {sub_id}", leave=False):
            read_sub_data(sub_id, task, db, file_key_list, total_seconds, rejected_seconds)


    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(file_key_list))
    txn.commit()
    db.close()

    hours = total_hours(total_seconds)
    print(f"Total raw duration across dataset: {hours:.2f} hours")
    rejected_hours = total_hours(rejected_seconds)
    pct_rejected = 100.0 * rejected_hours / hours

    print(f"Rejected (too short) duration: {rejected_hours:.2f} hours ({pct_rejected:.4f}% of total)")

        