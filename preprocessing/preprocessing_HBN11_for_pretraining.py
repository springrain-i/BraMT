import mne
from mne_bids import BIDSPath, read_raw_bids , get_entity_vals
import matplotlib.pyplot as plt
import os
import re

montage = mne.channels.make_standard_montage('GSN-HydroCel-129')

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

def data_process(raw : mne.io.Raw) -> mne.io.Raw:
    """Processes raw EEG data."""
    raw.filter(1.,100.)
    raw.notch_filter(60.)
    return raw

def read_sub_data(sub_id , task_name):
    """Reads raw EEG data for a given subject ID and task name."""
    print(f"Reading data for subject {sub_id}, task {task_name}")
    bids_path = BIDSPath(
        subject=sub_id,
        task=task_name,
        root=data_path,
    )
    bids_path = BIDSPath(root=data_path, subject=sub_id, task=task_name)
    all_paths = bids_path.match()
    runs = sorted(set(p.run for p in all_paths if p.run is not None))
    if not runs:
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.set_channel_types({ch: 'eeg' for ch in raw.info['ch_names']})
        raw.set_montage(montage)
        raw.drop_channels(['Cz'])
        raw = data_process(raw)
        raw.save(f'./data/sub-{sub_id}_task-{task_name}_eeg.fif', overwrite=True)
    else:
        for run in runs:
            bids_path_run = bids_path.copy().update(run=run)
            raw_run = read_raw_bids(bids_path=bids_path_run, verbose=False)
            raw_run.set_channel_types({ch: 'eeg' for ch in raw_run.info['ch_names']})
            raw_run.set_montage(montage)
            raw_run.drop_channels(['Cz'])
            raw_run = data_process(raw_run)
            raw_run.save(f'./data/sub-{sub_id}_task-{task_name}_run-{run}_eeg.fif', overwrite=True)

if __name__ == "__main__":
    data_path = '/media/ptiris/EXTERNAL_US/HBN-Release11'
    sub_id = 'NDARAB678VYW'
    tasks = read_sub_tasks(sub_id, data_path)
    print(tasks)
    read_sub_data(sub_id, tasks[0])
        