import os
import math
import datetime
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------------------------
# 2. Resampling with SimpleITK
# ---------------------------
def resample_image(itk_image, new_spacing=[3.27, 3.27, 3.27]):
    """
    Resample an ITK image to a given isotropic spacing.

    Parameters:
        itk_image (SimpleITK.Image): Input ITK image.
        new_spacing (list): Desired voxel spacing [sx, sy, sz].

    Returns:
        SimpleITK.Image: Resampled image.
    """
    original_spacing = itk_image.GetSpacing()  # Old spacing
    original_size = itk_image.GetSize()  # Old shape (number of voxels)
    
    # Compute new shape using the correct formula
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    
    print(f"Original Spacing: {original_spacing}")
    print(f"Original Size: {original_size}")
    print(f"New Spacing: {new_spacing}")
    print(f"Computed New Size: {new_size}")

    # Apply resampling
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetInterpolator(sitk.sitkLinear)

    resampled_image = resample.Execute(itk_image)
    return resampled_image


def parse_dicom_time(time_str):
    """
    Parses a DICOM time string in HHMMSS or HHMMSS.FFFFFF format.
    Raises a ValueError if parsing fails.
    """
    if not time_str:
        raise ValueError("Empty DICOM time string.")
    try:
        if '.' in time_str:
            return datetime.datetime.strptime(time_str, "%H%M%S.%f")
        else:
            return datetime.datetime.strptime(time_str, "%H%M%S")
    except Exception as e:
        raise ValueError(f"Error parsing time '{time_str}': {e}")


def parse_dicom_datetime(datetime_str):
    """
    Parses a DICOM datetime string in YYYYMMDDHHMMSS.FFFFFF format.
    Raises a ValueError if parsing fails.
    """
    if not datetime_str:
        raise ValueError("Empty DICOM datetime string.")
    try:
        # Split off any fractional seconds for parsing.
        return datetime.datetime.strptime(datetime_str.split('.')[0], "%Y%m%d%H%M%S")
    except Exception as e:
        raise ValueError(f"Error parsing datetime '{datetime_str}': {e}")


def calculate_decay_correction_factor(series_time, start_time, half_life):
    """
    Calculates the decay correction factor based on the time difference.
    
    Parameters:
      series_time (datetime): Time of series acquisition.
      start_time (datetime): Radiopharmaceutical injection time.
      half_life (float): Half-life in seconds.
      
    Returns:
      float: Decay correction factor.
      
    Raises:
      ValueError: If required times are missing or if half_life is not positive.
    """
    if series_time is None or start_time is None:
        raise ValueError("Series time or injection time is missing.")
    if half_life <= 0:
        raise ValueError("Half-life must be a positive value.")
    delta_t = (series_time - start_time).total_seconds()
    return math.exp(-math.log(2) * delta_t / half_life)


def extract_radiopharmaceutical_info(ds):
    """
    Extracts radiopharmaceutical metadata from the DICOM dataset.
    
    Returns:
      dict: Containing total_dose, half_life, and injection_time.
      
    Raises:
      ValueError: If any required attribute is missing or cannot be parsed.
    """
    if "RadiopharmaceuticalInformationSequence" not in ds:
        raise ValueError("Missing RadiopharmaceuticalInformationSequence in DICOM dataset.")
    
    radio_info = ds.RadiopharmaceuticalInformationSequence[0]  # Use the first entry

    if "RadionuclideTotalDose" not in radio_info:
        raise ValueError("Missing RadionuclideTotalDose in RadiopharmaceuticalInformationSequence.")
    if "RadionuclideHalfLife" not in radio_info:
        raise ValueError("Missing RadionuclideHalfLife in RadiopharmaceuticalInformationSequence.")

    total_dose = float(radio_info["RadionuclideTotalDose"].value)
    half_life = float(radio_info["RadionuclideHalfLife"].value)

    # Extract injection time from RadiopharmaceuticalStartDateTime or StartTime
    if "RadiopharmaceuticalStartDateTime" in radio_info:
        injection_time = parse_dicom_datetime(radio_info["RadiopharmaceuticalStartDateTime"].value)
    elif "RadiopharmaceuticalStartTime" in radio_info:
        injection_time = parse_dicom_time(radio_info["RadiopharmaceuticalStartTime"].value)
    else:
        raise ValueError("Missing both RadiopharmaceuticalStartDateTime and RadiopharmaceuticalStartTime.")
    
    return {"total_dose": total_dose, "half_life": half_life, "injection_time": injection_time}


def extract_series_time(ds):
    """
    Extracts the series acquisition time from DICOM metadata.
    Tries to use SeriesDate/SeriesTime; if SeriesDate is missing or invalid,
    falls back to AcquisitionDate/AcquisitionTime.
    
    Returns:
      datetime: Parsed Series Time.
      
    Raises:
      ValueError: If required date/time attributes are missing or cannot be parsed.
    """
    if "SeriesDate" in ds and "SeriesTime" in ds:
        series_date = ds.get("SeriesDate")
        series_time = ds.get("SeriesTime", "000000")
        # Ensure we extract the value if it's a DataElement
        if hasattr(series_date, "value"):
            series_date = series_date.value
        if hasattr(series_time, "value"):
            series_time = series_time.value
        if series_date == "19000101" and "AcquisitionDate" in ds:
            series_date = ds.get("AcquisitionDate")
            if hasattr(series_date, "value"):
                series_date = series_date.value
        try:
            return parse_dicom_datetime(series_date + series_time)
        except Exception as e:
            raise ValueError(f"Error parsing series acquisition datetime: {e}")
    else:
        raise ValueError("Missing SeriesDate and/or SeriesTime in DICOM metadata.")

def convert_to_suv(pixel_array, ds_list):
    """
    Converts raw pixel data from a PET series to an SUV image.
    
    Parameters:
      pixel_array (np.ndarray): 3D array of raw pixel values.
      ds_list (list): List of DICOM datasets (one per slice).
      
    Returns:
      np.ndarray: SUV image.
      
    Raises:
      ValueError: If any required DICOM attribute is missing or invalid.
    """
    print("---- SUV Conversion Debug Info ----")
    
    # Extract slice-specific RescaleSlope & RescaleIntercept (using .value for DataElements)
    rescale_slopes = []
    rescale_intercepts = []
    for ds in ds_list:
        if "RescaleSlope" not in ds:
            raise ValueError(f"Missing RescaleSlope in DICOM slice {ds.get('InstanceNumber', 'unknown')}.")
        if "RescaleIntercept" not in ds:
            raise ValueError(f"Missing RescaleIntercept in DICOM slice {ds.get('InstanceNumber', 'unknown')}.")
        slope = ds.get("RescaleSlope")
        intercept = ds.get("RescaleIntercept")
        # If these are DataElement objects, extract their values.
        if hasattr(slope, "value"):
            slope = slope.value
        if hasattr(intercept, "value"):
            intercept = intercept.value
        rescale_slopes.append(float(slope))
        rescale_intercepts.append(float(intercept))
    
    rescale_slopes = np.array(rescale_slopes)
    rescale_intercepts = np.array(rescale_intercepts)
    print("Max Rescale Slope  :", rescale_slopes.max())
    print("Mean Rescale Slope :", rescale_slopes.mean())
    print("Max Rescale Intercept:", rescale_intercepts.max())
    
    # Extract radiopharmaceutical info from the first slice (assuming consistency)
    info = extract_radiopharmaceutical_info(ds_list[0])
    total_dose = info["total_dose"]
    half_life = info["half_life"]
    injection_time = info["injection_time"]
    
    print("Radionuclide Total Dose (Bq):", total_dose)
    print("Radionuclide Half-Life (s):", half_life)
    print("Injection Time:", injection_time)
    
    # Extract series acquisition time
    series_time = extract_series_time(ds_list[0])
    print("Series Time:", series_time)
    
    # Extract Patient Weight, using .value if needed
    if "PatientWeight" not in ds_list[0]:
        raise ValueError("Missing PatientWeight in DICOM metadata of the first slice.")
    patient_weight = ds_list[0]["PatientWeight"]
    if hasattr(patient_weight, "value"):
        patient_weight = patient_weight.value
    patient_weight = float(patient_weight)
    print("Patient Weight (kg):", patient_weight)
    
    # Validate dose and half-life
    if total_dose <= 0:
        raise ValueError("Radionuclide Total Dose must be greater than zero.")
    if half_life <= 0:
        raise ValueError("Radionuclide Half-Life must be greater than zero.")
    
    decay_factor = calculate_decay_correction_factor(series_time, injection_time, half_life)
    print("Decay Correction Factor:", decay_factor)
    
    # Apply slopes and intercepts per slice (assuming pixel_array shape is [num_slices, H, W])
    activity_concentration = (pixel_array * rescale_slopes[:, None, None]) + rescale_intercepts[:, None, None]
    print("Max activity concentration (Bq/ml):", np.max(activity_concentration))
    
    corrected_dose = total_dose * decay_factor
    print("Corrected Dose (Bq):", corrected_dose)
    
    suv_image = (activity_concentration * patient_weight * 1000) / corrected_dose
    print("Max SUV value calculated:", np.max(suv_image))
    print("------------------------------------")
    
    return suv_image


# %%
def generate_mip(volume, view="sagittal"):
    """
    Generate a Maximum Intensity Projection (MIP) from a 3D PET SUV image.
    
    Parameters:
        volume (np.ndarray): 3D PET SUV image.
        view (str): 'axial', 'coronal', or 'sagittal'.
        clip_min (float): Minimum value for clipping (default: 0 SUV).
        clip_max (float): Maximum value for clipping (default: 30 SUV).

    Returns:
        np.ndarray: Normalized MIP image.
    """
    if view == "axial":
        mip = np.max(volume, axis=0)  # Max projection along Z-axis (axial)
    elif view == "coronal":
        mip = np.max(volume, axis=1)  # Max projection along Y-axis (coronal)
    elif view == "sagittal":
        mip = np.max(volume, axis=2)  # Max projection along X-axis (sagittal)
    else:
        raise ValueError("Invalid view. Choose from 'axial', 'coronal', or 'sagittal'.")
    
    return mip




# # %%
# def get_bbox(img, threshold=0.02):
#     """
#     Extracts the bounding box around the relevant non-zero pixels in the image.
#     Uses a threshold to exclude near-zero background pixels.
    
#     Parameters:
#         img (np.ndarray): Input 2D MIP image.
#         threshold (float): Minimum value to consider as "non-background".
        
#     Returns:
#         np.ndarray: Cropped image containing only the relevant region.
#     """
#     mask = img > threshold  # Create a mask where pixel values exceed threshold
    
#     if np.any(mask):  # If there are non-background pixels
#         rows = np.any(mask, axis=1)
#         cols = np.any(mask, axis=0)
#         rmin, rmax = np.where(rows)[0][[0, -1]]
#         cmin, cmax = np.where(cols)[0][[0, -1]]
#         return img[rmin:rmax+1, cmin:cmax+1]  # +1 to include last pixel
#     else:
#         return img  # Return original if no valid region found


# # %%
# def get_bbox_percentile(img, lower_percentile=5):
#     """
#     Extracts the bounding box by removing the lowest X% of pixel intensities.
    
#     Parameters:
#         img (np.ndarray): Input 2D MIP image.
#         lower_percentile (float): Percentile below which pixels are considered background.
    
#     Returns:
#         np.ndarray: Cropped image.
#     """
#     threshold = np.percentile(img, lower_percentile)  # Compute a dynamic threshold
#     return get_bbox(img, threshold)




def preprocess_pet_series(series_data, new_spacing=[3.27, 3.27, 3.27]):
    """
    Applies the full preprocessing pipeline to a single PET series.
        
    Parameters:
      series_data (dict): Dictionary with keys "volume" and "metadata_list" from a PET series.
      new_spacing (list): Target spacing for resampling (e.g., [4, 4, 4]).
      bbox_percentile (float): Lower percentile for bounding box extraction.
      
    Returns:
      dict: Containing:
            - "mip_coronal": Processed coronal MIP image after normalization and bounding box extraction.
            - "mip_sagittal": Processed sagittal MIP image after normalization and bounding box extraction.
            - "suv_volume": The full SUV-converted, resampled volume.
    """
    # Extract the raw volume and list of DICOM slices.
    raw_volume = series_data["volume"]
    ds_list = series_data["metadata_list"]

    # ---------------------------
    # Step A: Convert Raw Volume to SUV Volume
    # ---------------------------
    suv_volume = convert_to_suv(raw_volume, ds_list)
    print("SUV volume shape before resampling:", suv_volume.shape)

    # ---------------------------
    # Step B: Resample the SUV Volume
    # ---------------------------
    pixel_spacing = ds_list[0].PixelSpacing  # (X, Y) spacing in mm
    slice_thickness = ds_list[0].SliceThickness  # Z spacing in mm

    itk_image = sitk.GetImageFromArray(suv_volume)
    itk_image.SetSpacing((float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)))
    itk_image_resampled = resample_image(itk_image, new_spacing=new_spacing)
    suv_volume_resampled = sitk.GetArrayFromImage(itk_image_resampled)
    print("Resampled SUV Volume shape:", suv_volume_resampled.shape)

    # ---------------------------
    # Step C: Crop only if still necessary after resampling
    # ---------------------------
    if suv_volume_resampled.shape[0] > 310:
        print(f"Resampled volume still exceeds 310 slices ({suv_volume_resampled.shape[0]} slices). Cropping.")
        suv_volume_resampled = suv_volume_resampled[:310, :, :]  # Crop only if needed

    print("Final SUV Volume shape after optional cropping:", suv_volume_resampled.shape)

    # ---------------------------
    # Step D: Generate MIP Images
    # ---------------------------
    mip_coronal = generate_mip(suv_volume_resampled, view="coronal")
    mip_sagittal = generate_mip(suv_volume_resampled, view="sagittal")

    # ---------------------------
    # Step E: Normalize and Crop the MIP Images
    # ---------------------------
    # bbox_coronal = get_bbox_percentile(mip_coronal, lower_percentile=bbox_percentile)
    # bbox_sagittal = get_bbox_percentile(mip_sagittal, lower_percentile=bbox_percentile)

    # Return the processed images but not bounded !
    return {
        "mip_coronal": mip_coronal,
        "mip_sagittal": mip_sagittal,
        "suv_volume": suv_volume_resampled
    }

# %%
