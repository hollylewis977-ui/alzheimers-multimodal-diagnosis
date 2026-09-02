# Converted from notebook: Sus_3D_preprocessing_2D.ipynb

# %% [code cell]
!pip install antspyx
! pip install pydicom nibabel dicom2nifti

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

# Unmount the drive first
drive.flush_and_unmount()
print('Drive unmounted')

   # Now mount it again
drive.mount('/content/Shareddrive')

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

# Step 5: Normalization
def normalize_image(img):
    img_array = img.numpy()
    normalized_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    normalized_image = ants.from_numpy(normalized_array, spacing=img.spacing)
    return normalized_image

# Step 6: Downsampling
def downsample_image(img):
    downsampled_image = ants.resample_image(img, (64,64,64), use_voxels=True)
    return downsampled_image

# Function to visualize one slice from each plane
def visualization(img):
    # Get the dimensions of the image
    depth, height, width = img.shape

    # Extract one slice from each plane
    sagittal_slice_index = depth // 2  # Middle sagittal slice
    coronal_slice_index = height // 2   # Middle coronal slice
    axial_slice_index = width // 2      # Middle axial slice

    # Extract slices
    sagittal_slice = img.numpy()[sagittal_slice_index, :, :]
    coronal_slice = img.numpy()[:, coronal_slice_index, :]
    axial_slice = img.numpy()[:, :, axial_slice_index]

    # Visualize each slice
    slices = {'sagittal': sagittal_slice, 'coronal': coronal_slice, 'axial': axial_slice}
    for plane, slice_to_display in slices.items():
        plt.imshow(slice_to_display, cmap='gray')
        plt.title(f'{plane.capitalize()} Slice')
        plt.axis('off')
        plt.show()

# %% [code cell]
# Define the root directory path
root_directory = '/content/drive/MyDrive/ML_CN_Data/ADNI'

# %% [code cell]
def process_and_convert_folders(directory, output_folder):
    # Check if the root directory exists
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    # Loop through items in the current directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)  # Get the full path

        # If the item is a directory, recurse into it
        if os.path.isdir(item_path):
            print(f"Entering directory: {item_path}")
            process_and_convert_folders(item_path, output_folder)  # Recurse into the subdirectory

    # Check for directories containing .dcm files
    dicom_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".dcm")]

    if len(dicom_files) > 0:  # Process if at least one DICOM file is found
        print(f"Found directory with {len(dicom_files)} DICOM file(s): {directory}")
        try:
            # Ensure the output folder exists
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Convert DICOM to NIfTI
            dicom2nifti.convert_dir.convert_directory(directory, output_folder, compression=True, reorient=True)
            print(f"Converted DICOM files in {directory} to NIfTI format.")
        except Exception as e:
            print(f"Error converting DICOM files in {directory}: {e}")
    else:
        print(f"No DICOM files found in directory: {directory}")

# Define the root directory and output folder
root_directory = '/content/drive/MyDrive/ML_CN_Data/ADNI'
output_folder = '/content/drive/MyDrive/ML_CN_Data/converted'

# Process the root directory
process_and_convert_folders(root_directory, output_folder)

# %% [code cell]
import os
import ants  # Ensure that the ANTs library is imported

# Define the root directory
root_directory = r'/content/mydrive/MyDrive/ML_Data/3D_datasets/CN_ADNI'
saved_directory = r"/content/mydrive/MyDrive/ML_Data/CN/extensive_preprocessed"

count = 0

def apply_pipeline(gz_path):
    # Extract filename without extension
    filename = os.path.basename(gz_path).replace(".gz", "")

    # Load, preprocess, and save the file
    try:
        # Step 1: Load images
        img_MRI_resampled, img_template_resampled, brain_mask_image = loading_ims(
            img=gz_path,
            template="MNI152_T1_1mm.nii",
            brain_mask="MNI152_T1_1mm_Brain_Mask.nii"
        )

        # Step 2: Registration
        aligned_image = regestration(img_MRI_resampled, img_template_resampled)

        # Step 3: Brain Masking
        brain_only_image = brain_masking(aligned_image, brain_mask_image)

        # Step 4: Winsorizing and Bias Field Correction
        preprocessed_image = winsorizing_biasfield(brain_only_image)

        #step 5:

        # Save the preprocessed image
        save_path = os.path.join(saved_directory, f"{filename}_preprocessed.nii")
        ants.image_write(preprocessed_image, save_path)
        print(f"Saved preprocessed image: {save_path}")

    except Exception as e:
        print(f"Error processing {gz_path}: {e}")

# Recursive function to navigate directories and process .dcm files
def process_folders(directory):
    global count  # Declare `count` as global to modify it inside the function
    # Loop through items in the current directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)  # Get the full path

        # Check if the item is a directory
        if os.path.isdir(item_path):
            print(f"Entering directory: {item_path}")
            process_folders(item_path)  # Recurse into the directory

        # If the item is not a directory, check for nifiti files
        elif os.path.isfile(item_path) and item_path.endswith('.nii'):
            print(f"Processing file: {item_path}")
            apply_pipeline(item_path)  # Apply the pipeline and save results
            count += 1  # Increment the counter

# Start processing from the root directory
process_folders(root_directory)
print('Number of images preprocessed:', count)

# %% [code cell]
# Global counter for processed images
count = 0

# Function to save one slice from each plane
def save_slices(img, output_folder, filename):
    # Get the dimensions of the image
    depth, height, width = img.shape

    # Extract one slice from each plane
    sagittal_slice_index = depth // 2  # Middle sagittal slice
    coronal_slice_index = height // 2   # Middle coronal slice
    axial_slice_index = width // 2      # Middle axial slice

    # Extract slices
    sagittal_slice = img.numpy()[sagittal_slice_index, :, :]
    coronal_slice = img.numpy()[:, coronal_slice_index, :]
    axial_slice = img.numpy()[:, :, axial_slice_index]

    # Save each slice as a .png file
    slices = {'sagittal': sagittal_slice, 'coronal': coronal_slice, 'axial': axial_slice}
    for plane, slice_to_save in slices.items():
        output_path = os.path.join(output_folder, f"{filename}_{plane}_slice.png")
        plt.imsave(output_path, slice_to_save, cmap='gray')
        print(f"{plane.capitalize()} slice saved at: {output_path}")

# Recursive function to navigate directories and process .nii files
def process_folders(directory, output_folder):
    global count  # Declare `count` as global to modify it inside the function
    # Loop through items in the current directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)  # Get the full path

        # Check if the item is a directory
        #if os.path.isdir(item_path):
            #print(f"Entering directory: {item_path}")
            #process_folders(item_path, output_folder)  # Recurse into the directory

        # If the item is not a directory, check for .nii files
        if os.path.isfile(item_path) and item_path.endswith('.nii'):
            print(f"Processing file: {item_path}")
            # Load the 3D image
            img = ants.image_read(item_path)

            # Prepare output filename (without extension)
            filename = os.path.splitext(os.path.basename(item_path))[0]

            # Save the slices
            save_slices(img, output_folder, filename)
            count += 1  # Increment the counter


# Define the root directory and output folder
root_directory = r'/content/Shareddrive/Shareddrives/ML_Data/3D_datasets/MCI/Preprocessed'
output_folder = r'/content/Shareddrive/Shareddrives/ML_Data/2D_datasets/MCI'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Start processing from the root directory
process_folders(root_directory, output_folder)
print('Number of images sliced:', count)