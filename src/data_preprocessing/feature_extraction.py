import pandas as pd
import cv2
import numpy as np
import os
import glob
from natsort import natsorted
from scipy.ndimage import find_objects
import mat73
import skimage.measure as measure
import matplotlib.pyplot as plt 


def cir(area, perimeter):
    if area == 0:
        return 0
    return (perimeter ** 2) / area

def dm(area, average_opd):

    alpha = 0.2
    if area == 0 or average_opd == 0:
        return 0
    return (area * 0.275 * 10**-12 * abs(average_opd * 0.275 * 10**-9)) / (alpha * 10**-18)

def pv(area, average_opd):
    alpha = 0.2
    dm2 = dm(area, average_opd)
    return dm2 * alpha

def psa(img, area):

    gray = img.astype('float32')
    gX_Scharr = cv2.Scharr(gray, ddepth=cv2.CV_32F, dx=1, dy=0)
    gY_Scharr = cv2.Scharr(gray, ddepth=cv2.CV_32F, dx=0, dy=1)
    gX_Scharr = cv2.convertScaleAbs(gX_Scharr)
    gY_Scharr = cv2.convertScaleAbs(gY_Scharr)
    inner = np.sqrt(1 + (gX_Scharr[:] * 10**-9)**2 + (gY_Scharr[:] * 10**-9)**2)
    return np.sum(inner[:]) + (area * 0.275**2)

def psa2dm(img, area, average_opd):
    psa1 = psa(img, area)
    dm1 = dm(area, average_opd)
    if dm1 == 0:
        return 0
    return psa1 / dm1

def a2v(area, average_opd):
    pv1 = pv(area, average_opd)
    if pv1 == 0:
        return 0
    return area * 0.275**2 / pv1

def speri(img, area, average_opd):

    pv1 = pv(area, average_opd)
    psa1 = psa(img, area)
    if psa1 == 0:
        return 0
    return (np.pi)**0.33 * ((6 * pv1)**0.667 / psa1)

def energy(img):
    
    img2 = img**2
    return 10**-6 * np.sum(img2[:])

def ellipticity(ma, mi):
    if (ma + 0.000001) == 0:
        return 0
    return mi / (ma + 0.000001)

def extract_features(mask_path, opd_path, output_folder, pixels_to_um=5.5/20):
    """
    Args:
        mask_path (str): Path to the directory containing mask .npy files.
        opd_path (str): Path to the directory containing OPD .mat files.
        output_folder (str): Path to the directory where output CSVs will be saved.
        pixels_to_um (float): Conversion factor from pixels to micrometers.
    """
    mask_names = natsorted(os.listdir(mask_path))
    opd_files = natsorted(os.listdir(opd_path))

    os.makedirs(output_folder, exist_ok=True)

    for mask_name, opd_file in zip(mask_names, opd_files):
        mask_file_path = os.path.join(mask_path, mask_name)
        opd_file_path = os.path.join(opd_path, opd_file)

        print(f"Processing {mask_name} and {opd_file}...")

        try:
            dat = np.load(mask_file_path, allow_pickle=True).item()
            mask = dat['masks']
            f11 = mat73.loadmat(opd_file_path)
            image = f11['opd_same']
            mask, num_objects = measure.label(mask, return_num=True)
            regions = measure.regionprops(mask, intensity_image=image)

            circularity = []
            dry_mass = []
            phase_surface_area = []
            psa2drymass = []
            sphericity = []
            variances = []
            skewness = []
            kurtosis = []
            energy_texture = []
            ellipse = []

            for region in regions:
                perimeter = region.perimeter
                area = region.area
                mean_intensity = region.intensity_mean
                img_intensity = region.image_intensity
                major_axis = region.axis_major_length
                minor_axis = region.axis_minor_length

                # Calculate custom features
                circularity.append(cir(area, perimeter))
                dry_mass.append(dm(area, mean_intensity))
                phase_surface_area.append(psa(img_intensity, area))
                psa2drymass.append(psa2dm(img_intensity, area, mean_intensity))
                sphericity.append(speri(img_intensity, area, mean_intensity))
                energy_texture.append(energy(img_intensity))
                ellipse.append(ellipticity(major_axis, minor_axis))

                try:
                    variances.append(region.moments_central[2, 0] * 0.275 / (region.moments_central[0, 0] + 1e-9)) # Add small epsilon to avoid division by zero
                except IndexError:
                    variances.append(np.nan)
                try:
                    skewness.append(region.moments_normalized[3, 0])
                except IndexError:
                    skewness.append(np.nan)

                try:
                    kurtosis.append(region.moments_hu[3])
                except IndexError:
                    kurtosis.append(np.nan)


            df_props = pd.DataFrame([
                {
                    'area': r.area * pixels_to_um**2,
                    'equivalent_diameter': r.equivalent_diameter_area * pixels_to_um,
                    'Major axis': r.axis_major_length * pixels_to_um,
                    'Minor axis': r.axis_minor_length * pixels_to_um,
                    'perimeter': r.perimeter * pixels_to_um,
                    'Min Intensity': r.intensity_min,
                    'Mean Intensity': r.intensity_mean,
                    'Max Intensity': r.intensity_max,
                    'Eccentricity': r.eccentricity
                } for r in regions
            ])

            df_handcrafted = pd.DataFrame({
                'Circularity': circularity,
                'Ellipticity': ellipse,
                'dry_mass': dry_mass,
                'PSA': phase_surface_area,
                'PSA2DM': psa2drymass,
                'Sphericity': sphericity,
                'Energy': energy_texture,
                'Variance': variances,
                'Kurtosis': kurtosis,
                'Skewness': skewness
            })

            df_final = pd.concat([df_props, df_handcrafted], axis=1)

            # Output CSV filename
            output_filename = os.path.splitext(mask_name)[0] + '.csv'
            output_path = os.path.join(output_folder, output_filename)

            df_final.to_csv(output_path, index=False)
            print(f"Saved features for {mask_name} to {output_path}")

        except Exception as e:
            print(f"Error processing {mask_name} or {opd_file}: {e}")

if __name__ == "__main__":
    mask_dir = "D:/Skin Cancer/dataset/masks"
    opd_dir = "D:/Skin Cancer/dataset/opd_values"
    output_feat_folder = "D:/Skin Cancer/dataset/Properties"

    # Feature extraction
    extract_features(mask_dir, opd_dir, output_feat_folder)
