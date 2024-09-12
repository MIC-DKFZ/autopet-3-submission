import os
import sys
import SimpleITK as sitk
import numpy as np

in_dir = "imagesTr"
lbl_dir = "labelsTr"
out_dir = "labelsTr_totalseg"
combined_dir = "labelsTr_organs"

lbl_mapping_all = {
    0: 0,
    1: 1, # Spleen
    2: 2, # Right kidney
    3: 2, # Left kidney
    5: 3, # Liver
    21: 4, # Urinary bladder
    10: 5, # Lung
    11: 5, # Lung
    12: 5, # Lung
    13: 5, # Lung
    14: 5, # Lung
    90: 6, # Brain
    51: 7, # Heart
    6: 8, # Stomach
    22: 9, # Prostate
}

lbl_mapping_head = {
    6: 10, 
    7: 10,
    8: 10,
    9: 10,
}

if __name__ == "__main__":
    cases = os.listdir(in_dir)
    cases = [case for case in cases if case.endswith("0000.nii.gz")]
    for case in cases:
        case_path = os.path.join(in_dir, case)
        out_path_all = os.path.join(out_dir, case.split("_0000")[0] + "_all.nii.gz")
        out_path_head = os.path.join(out_dir, case.split("_0000")[0] + "_head.nii.gz")

        # Check if the output files already exist
        if os.path.exists(out_path_all) and os.path.exists(out_path_head):
            continue

        command = f"TotalSegmentator -i '{case_path}' -o '{out_path_all}' --ml"
        os.system(command)
        command = f"TotalSegmentator -i '{case_path}' -o '{out_path_head}' -ta head_glands_cavities --ml"
        os.system(command)

        # Load the total segmentation labels
        lbl_all_img = sitk.ReadImage(out_path_all)
        lbl_all = sitk.GetArrayFromImage(lbl_all_img)

        # Load the head segmentation labels
        lbl_head = sitk.ReadImage(out_path_head)
        lbl_head = sitk.GetArrayFromImage(lbl_head)

        # Combine the labels
        lbl_combined = np.zeros_like(lbl_head)
        for key in lbl_mapping_all:
            lbl_combined[lbl_all == key] = lbl_mapping_all[key]
        for key in lbl_mapping_head:
            lbl_combined[lbl_head == key] = lbl_mapping_head[key]

        # Save the combined labels
        lbl_combined_img = sitk.GetImageFromArray(lbl_combined)
        lbl_combined_img.CopyInformation(lbl_all_img)
        out_path_combined = os.path.join(combined_dir, case.split("_0000")[0] + ".nii.gz")
        sitk.WriteImage(lbl_combined_img, out_path_combined)

        print(f"Finished processing {case_path}")
