# Converted from notebook: Preprocessing.ipynb

# %% [code cell]

# %% [code cell]
import ants
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import nibabel as nib
import dicom2nifti
import os

# %% [code cell]
from google.colab import drive
drive.mount('/content/drive')
import os

# %% [code cell]
def loading_ims(img, template= "MNI152_T1_1mm.nii", brain_mask=  "MNI152_T1_1mm_Brain_Mask.nii"):
    mri_image = ants.image_read(img, reorient='IAL')
    template_image = ants.image_read(template, reorient='IAL')
    brain_mask = ants.image_read(brain_mask, reorient='IAL')

    img_MRI_resampled = ants.resample_image(mri_image, (1, 1, 1.2), False, 4)
    img_template_resampled = ants.resample_image(template_image, (1, 1, 1.2), False, 4)
    return img_MRI_resampled, img_template_resampled, brain_mask

def regestration(img_MRI_resampled, img_template_resampled):
    affine_registration = ants.registration(fixed=img_template_resampled, moving=img_MRI_resampled, type_of_transform="Affine")
    img_MRI_resampled_aligned = affine_registration['warpedmovout']
    registration = ants.registration(fixed=img_template_resampled, moving=img_MRI_resampled_aligned, type_of_transform="SyN")
    aligned_image = registration['warpedmovout']
    return aligned_image

def brain_masking(aligned_image, brain_mask):
    brain_mask = ants.resample_image_to_target(brain_mask, aligned_image)
    brain_only_image = aligned_image * brain_mask
    return brain_only_image

def winsorizing_biasfield(brain_only_image):
    winsorized = ants.iMath(brain_only_image, "TruncateIntensity", 0.05, 0.95)
    n4_corrected_image = ants.n4_bias_field_correction(winsorized)
    return n4_corrected_image


def normalize_image(img):
    img_array = img.numpy()
    normalized_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    normalized_image = ants.from_numpy(normalized_array, spacing=img.spacing)
    return normalized_image


def downsample_image(img):
    downsampled_image = ants.resample_image(img, (64,64,64), use_voxels=True)
    return downsampled_image


def visualization(img):
    slice_index = img.shape[2] // 2
    slice_to_display = img.numpy()[:, :, slice_index]
    plt.imshow(slice_to_display, cmap='gray')
    plt.title('Image Visualization')
    plt.show()
    print("Image visualization completed.")

# %% [code cell]
#@title Cobvering Dicom to Niftii
import os
import dicom2nifti

root_directory = '/content/drive/MyDrive/MachineLearning_AD_Project/Data/MCI/ADNI'

# Recursive function to navigate directories and process directories with multiple .dcm files
def process_and_convert_folders(directory, output_folder):
    # loop through items in the current directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)  # Get the full path

        # If the item is a directory, recurse into it
        if os.path.isdir(item_path):
            process_and_convert_folders(item_path, output_folder)  # recurse into the subdirectory

    dicom_files = []  # Collect all .dcm files in the directory
    for file in os.listdir(directory):
        if file.endswith(".dcm"):
            dicom_files.append(os.path.join(directory, file))

    if len(dicom_files) > 1:  # Process only if multiple DICOM files are found
        dicom2nifti.convert_dir.convert_directory(directory, output_folder, compression=True, reorient=True)

output_folder = '/content/drive/MyDrive/MachineLearning_AD_Project/Data/MCI/Converted'
process_and_convert_folders(root_directory, output_folder)

# %% [code cell]
#@title preprocessing the converted images and saving them
import os
import ants

root_directory = r'/content/drive/MyDrive/MachineLearning_AD_Project/Data/MCI/Converted'
saved_directory = r"/content/drive/MyDrive/MachineLearning_AD_Project/Data/MCI/preprocessed"


def apply_pipeline(img_path):
    img_MRI_resampled, img_template_resampled, brain_mask_image = loading_ims(
            img= img_path,
            template="MNI152_T1_1mm.nii",
            brain_mask="MNI152_T1_1mm_Brain_Mask.nii"
        )

    aligned_image = regestration(img_MRI_resampled, img_template_resampled)
    brain_only_image = brain_masking(aligned_image, brain_mask_image)
    preprocessed_image = winsorizing_biasfield(brain_only_image)
    save_path = os.path.join(saved_directory, f"{filename}_preprocessed.nii")
    ants.image_write(preprocessed_image, save_path)

def process_folders(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path): # if this directory not file
            process_folders(item_path)

        # if the item is not a directory, check for .dcm files
        elif os.path.isfile(item_path) and item_path.endswith(".nii"):
            apply_pipeline(item_path)

process_folders(root_directory)