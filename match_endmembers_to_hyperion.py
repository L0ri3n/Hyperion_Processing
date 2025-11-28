"""
Match endmember library wavelengths to preprocessed Hyperion wavelengths
The preprocessed Hyperion has removed bad bands (195 bands instead of 242)
"""

import numpy as np
import pandas as pd
from spectral import envi

print("=" * 70)
print("Matching Endmember Library to Hyperion Wavelengths")
print("=" * 70)

# Load Hyperion image to get actual wavelengths
hyperion_path = './amd_mapping/data/hyperion/EO1H2020342016359110KF_reflectance.hdr'
print("\n1. Loading Hyperion wavelengths...")
img = envi.open(hyperion_path)
hyperion_wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
print("   Hyperion bands: {}".format(len(hyperion_wavelengths)))
print("   Wavelength range: {:.1f} - {:.1f} nm".format(
    hyperion_wavelengths.min(), hyperion_wavelengths.max()))

# Load full endmember library
library_path = './amd_mapping/data/outputs/test_endmember_library.csv'
print("\n2. Loading endmember library...")
library_df = pd.read_csv(library_path, index_col=0)
library_wavelengths = library_df.index.values
print("   Library bands: {}".format(len(library_wavelengths)))
print("   Wavelength range: {:.1f} - {:.1f} nm".format(
    library_wavelengths.min(), library_wavelengths.max()))

# Resample each endmember to Hyperion wavelengths
print("\n3. Resampling endmembers to Hyperion wavelengths...")
resampled_endmembers = {}

for mineral_name in library_df.columns:
    spectrum = library_df[mineral_name].values
    # Interpolate to Hyperion wavelengths
    resampled = np.interp(hyperion_wavelengths, library_wavelengths, spectrum)
    resampled_endmembers[mineral_name] = resampled
    print("   Resampled: {}".format(mineral_name))

# Create new DataFrame with matched wavelengths
matched_df = pd.DataFrame(resampled_endmembers, index=hyperion_wavelengths)

# Save matched library
output_path = './amd_mapping/data/outputs/endmember_library_matched.csv'
matched_df.to_csv(output_path)
print("\n4. Saved matched library to: {}".format(output_path))
print("   New shape: {} bands × {} minerals".format(len(matched_df), len(matched_df.columns)))

print("\n" + "=" * 70)
print("Done! Endmember library now matches Hyperion wavelengths.")
print("=" * 70)
