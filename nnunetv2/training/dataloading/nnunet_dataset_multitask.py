import os
from typing import List

import numpy as np
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile
from nnunetv2.training.dataloading.utils import get_case_identifiers


class nnUNetDatasetMultiTask(object):
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):
        """
        This does not actually load the dataset. It merely creates a dictionary where the keys are training case names and
        the values are dictionaries containing the relevant information for that case.
        dataset[training_case] -> info
        Info has the following key:value pairs:
        - dataset[case_identifier]['properties']['data_file'] -> the full path to the npz file associated with the training case
        - dataset[case_identifier]['properties']['properties_file'] -> the pkl file containing the case properties

        In addition, if the total number of cases is < num_images_properties_loading_threshold we load all the pickle files
        (containing auxiliary information). This is done for small datasets so that we don't spend too much CPU time on
        reading pkl files on the fly during training. However, for large datasets storing all the aux info (which also
        contains locations of foreground voxels in the images) can cause too much RAM utilization. In that
        case is it better to load on the fly.

        If properties are loaded into the RAM, the info dicts each will have an additional entry:
        - dataset[case_identifier]['properties'] -> pkl file content

        IMPORTANT! THIS CLASS ITSELF IS READ-ONLY. YOU CANNOT ADD KEY:VALUE PAIRS WITH nnUNetDataset[key] = value
        USE THIS INSTEAD:
        nnUNetDataset.dataset[key] = value
        (not sure why you'd want to do that though. So don't do it)
        """
        super().__init__()
        # print('loading dataset')
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {}
            self.dataset[c]['data_file'] = join(folder, f"{c}.npz")
            self.dataset[c]['properties_file'] = join(folder, f"{c}.pkl")
            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, f"{c}.npz")

        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for i in self.dataset.keys():
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])

        self.keep_files_open = ('nnUNet_keep_files_open' in os.environ.keys()) and \
                               (os.environ['nnUNet_keep_files_open'].lower() in ('true', '1', 't'))
        # print(f'nnUNetDataset.keep_files_open: {self.keep_files_open}')

    def __getitem__(self, key):
        ret = {**self.dataset[key]}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret

    def __setitem__(self, key, value):
        return self.dataset.__setitem__(key, value)

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return self.dataset.__len__()

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    def load_case(self, key):
        entry = self[key]
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + ".npy"):
            data = np.load(entry['data_file'][:-4] + ".npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
                # print('saving open data file')
        else:
            data = np.load(entry['data_file'])['data']

        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
                # print('saving open seg file')
        else:
            seg = np.load(entry['data_file'])['seg']


        if isfile(entry['data_file'][:-4] + "_seg_org.npy"):
            seg_prev = np.load(entry['data_file'][:-4] + "_seg_org.npy", 'r')
        else:
            raise ValueError(f"Could not find the organ segmentation {entry['data_file'][:-4] + '_seg_org.npy'}. Please preprocess and unpack the organ "
                             "segmentations and place them in the same folder as the data files with the ending '[name]_seg_org.npy'. Double check if your plans file points to the correct dataset and preprocessed folder.")
        seg = np.vstack((seg, seg_prev)) # [None] removed

        return data, seg, entry['properties']
    
    def load_case_validation(self, key):
        entry = self[key]
        if 'open_data_file' in entry.keys():
            data = entry['open_data_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + ".npy"):
            data = np.load(entry['data_file'][:-4] + ".npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_data_file'] = data
                # print('saving open data file')
        else:
            data = np.load(entry['data_file'])['data']

        if 'open_seg_file' in entry.keys():
            seg = entry['open_seg_file']
            # print('using open data file')
        elif isfile(entry['data_file'][:-4] + "_seg.npy"):
            seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
            if self.keep_files_open:
                self.dataset[key]['open_seg_file'] = seg
                # print('saving open seg file')
        else:
            seg = np.load(entry['data_file'])['seg']

        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                seg_prev = np.load(entry['seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                seg_prev = np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((seg, seg_prev[None]))

        return data, seg, entry['properties']

