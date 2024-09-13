import os
import shutil
from nnunetv2.training.dataloading.utils import unpack_dataset
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.utilities.file_path_utilities import maybe_convert_to_dataset_name
from tqdm import tqdm

def merge_lesions_and_organ_dataset(lesion_dataset_id, organ_dataset_id):
    lesion_dataset = maybe_convert_to_dataset_name(lesion_dataset_id)
    organ_dataset = maybe_convert_to_dataset_name(organ_dataset_id)
    print(f"Merging organ dataset {organ_dataset} into lesion dataset {lesion_dataset}")

    print("Unpacking both datasets. This can take some time...")
    unpack_dataset(join(nnUNet_preprocessed, lesion_dataset, "nnUNetPlans_3d_fullres"), unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)), verify_npy=True)
    unpack_dataset(join(nnUNet_preprocessed, organ_dataset, "nnUNetPlans_3d_fullres"), unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)), verify_npy=True)
    
    print("Merging datasets...")
    organ_seg_files = [f for f in os.listdir(join(nnUNet_preprocessed, organ_dataset, "nnUNetPlans_3d_fullres")) if f.endswith("_seg.npy")]
    
    for f in tqdm(organ_seg_files):
        file_path = join(nnUNet_preprocessed, organ_dataset, "nnUNetPlans_3d_fullres", f)
        save_path = join(nnUNet_preprocessed, lesion_dataset, "nnUNetPlans_3d_fullres", f.replace("_seg.npy", "_seg_org.npy"))
        if not os.path.isfile(save_path):
            shutil.copy(file_path, save_path)

    print("Done merging datasets.")


def merge_datasets_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lesion_dataset_id', type=int, required=True, help='dataset id of the lesion dataset')
    parser.add_argument('-o', '--organ_dataset_id', type=int, required=True, help='dataset id of the organ dataset')
    args = parser.parse_args()
    merge_lesions_and_organ_dataset(args.lesion_dataset_id, args.organ_dataset_id)


if __name__ == "__main__":
    merge_lesions_and_organ_dataset(610, 612)