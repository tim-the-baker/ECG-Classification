import os
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from exploration.directories import ECG_DIR, LABEL_DIR

# given a heart rhythm abbreviation, this dict returns the rhythm's full name.
RHYTHM_ABBR_TO_FULL = {
    "AFb": "Atrial Fibrillation",
    "AFt": "Atrial Flutter",
    "SR": "Sinus Rhythm",
    "SVT": "Supraventricular Tachycardia",
    "VFb": "Ventricular Fibrillation",
    "VFt": "Ventricular Flutter",
    "VPD": "Ventricular Premature Depolarizations",
    "VT": "Ventricular Tachycardia",
}

RHYTHM_ABBR_TO_LABEL = {
    "AFb": 0,
    "AFt": 1,
    "SR": 2,
    "SVT": 3,
    "VFb": 4,
    "VFt": 5,
    "VPD": 6,
    "VT": 7,
}

LABEL_TO_RHYTHM_ABBR = {
    0: "AFb",
    1: "AFt",
    2: "SR",
    3: "SVT",
    4: "VFb",
    5: "VFt",
    6: "VPD",
    7: "VT",
}

# the ECG signal data are sampled at 250 Hz and are 5 seconds in length. Thus, the signal length is 1250 samples.
ECG_LENGTH = 1250

def create_multiclass_label_file():
    """
    The TinyML contest only provided a label file for binary classification because that was the focus of the competition.
    However, based on the ECG file names, we can infer the multiclass rhythm label. This function creates a label file
    for multiclass classification for later experiments. This function only needs to be run once
    :return: None
    """
    for mode in ["train", "test"]:
        file_string = "label,filename"
        reference_file = os.path.join(LABEL_DIR, f"{mode}_indices.csv")
        new_file = os.path.join(LABEL_DIR, f"{mode}_indices_multiclass.csv")
        with open(reference_file, 'r') as f:
            f.readline()
            for line in f.readlines():
                ecg_filename = line.split(',')[1][:-1]
                found = False
                for word in ecg_filename.split('-'):
                    for rhythm in RHYTHM_ABBR_TO_LABEL.keys():
                        if rhythm == word:
                            if found:
                                print(f"ERROR ON {ecg_filename}.")
                            file_string += f"\n{RHYTHM_ABBR_TO_LABEL[rhythm]},{ecg_filename}"
                            found = True
                if not found:
                    print(f"ERROR ON {ecg_filename}.")
        with open(new_file, 'w') as f:
            f.write(file_string)


class IEGM_Dataset(Dataset):
    def __init__(self, train, binary_classification, ecg_dir=ECG_DIR, label_dir=LABEL_DIR, ecg_length=ECG_LENGTH,
                 transform=None, target_transform=None):
        self.train = train
        self.binary_classification = binary_classification
        self.ecg_dir = ecg_dir
        self.label_dir = label_dir
        self.ecg_length = ecg_length
        self.transform = transform
        self.target_transform = target_transform

        set = "train" if train else "test"
        multi = "" if binary_classification else "_multiclass"
        self.label_file = os.path.join(self.label_dir, f"{set}_indices{multi}.csv")
        self.ecg_labels = pd.read_csv(self.label_file)

    def __len__(self):
        return len(self.ecg_labels)

    def __getitem__(self, idx):
        ecg_file = os.path.join(self.ecg_dir, self.ecg_labels.iloc[idx, 1])
        ecg = pd.read_csv(ecg_file, header=None).squeeze("columns").to_numpy()
        label = self.ecg_labels.iloc[idx, 0]
        if self.transform:
            ecg = self.transform(ecg)
        if self.target_transform:
            label = self.target_transform(label)
        return ecg, label

