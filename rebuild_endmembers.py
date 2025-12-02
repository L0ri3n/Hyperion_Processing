"""
Script to rebuild endmember library in correct SNAP-compatible format
"""

import numpy as np
import pandas as pd
import os
import sys
from spectral import envi

# Import the corrected save functions
from hyperion_workflow import save_envi_library, validate_envi_library, save_envi_library_alternative

def rebuild_endmember_library():
    """
    Read the existing CSV endmember library and rebuild as proper ENVI spectral library
    """
    print("="*70)
    print("REBUILDING ENDMEMBER LIBRARY FOR SNAP COMPATIBILITY")
    print("="*70)

    # Paths
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'amd_mapping', 'data', 'outputs')

    csv_file = os.path.join(data_dir, 'endmember_library.csv')
    output_file = os.path.join(data_dir, 'endmember_library_fixed')

    # Check if CSV exists
    if not os.path.exists(csv_file):
        print("\n[ERROR] CSV file not found: {}".format(csv_file))
        return False

    print("\n1. Loading endmember data from CSV...")
    print("   {}".format(csv_file))

    # Load CSV
    try:
        df = pd.read_csv(csv_file, index_col=0)
        print("   [OK] Loaded CSV with {} minerals and {} bands".format(
            len(df.columns), len(df.index)))
    except Exception as e:
        print("   [ERROR] Failed to load CSV: {}".format(str(e)))
        return False

    # Extract wavelengths and endmembers
    wavelengths = df.index.values.astype(float)
    endmembers = {}

    print("\n2. Extracting mineral spectra:")
    for col in df.columns:
        spectrum = df[col].values.astype(np.float32)
        endmembers[col] = spectrum
        print("   - {}: {} bands, range [{:.4f}, {:.4f}]".format(
            col, len(spectrum), np.min(spectrum), np.max(spectrum)))

    # Save with corrected format
    print("\n3. Saving ENVI spectral library (corrected format)...")
    try:
        save_envi_library(endmembers, wavelengths, output_file)
        print("   [OK] Saved to: {}".format(output_file))
    except Exception as e:
        print("   [ERROR] Save failed: {}".format(str(e)))
        return False

    # Validate
    print("\n4. Validating corrected library...")
    hdr_file = output_file + '.hdr'
    if validate_envi_library(hdr_file):
        print("   [SUCCESS] Library is SNAP-compatible!")

        # Also save the matched library if it exists
        csv_matched = os.path.join(data_dir, 'endmember_library_matched.csv')
        if os.path.exists(csv_matched):
            print("\n5. Also rebuilding matched library...")
            try:
                df_matched = pd.read_csv(csv_matched, index_col=0)
                wavelengths_matched = df_matched.index.values.astype(float)
                endmembers_matched = {}

                for col in df_matched.columns:
                    endmembers_matched[col] = df_matched[col].values.astype(np.float32)

                output_matched = os.path.join(data_dir, 'endmember_library_matched_fixed')
                save_envi_library(endmembers_matched, wavelengths_matched, output_matched)
                print("   [OK] Saved matched library")

                if validate_envi_library(output_matched + '.hdr'):
                    print("   [SUCCESS] Matched library is SNAP-compatible!")
                    return True
            except Exception as e:
                print("   [WARNING] Matched library rebuild failed: {}".format(str(e)))

        return True
    else:
        print("   [FAILED] Validation failed")
        print("\n5. Trying alternative save method...")
        try:
            save_envi_library_alternative(endmembers, wavelengths, output_file + '_alt')
            if validate_envi_library(output_file + '_alt.hdr'):
                print("   [SUCCESS] Alternative method worked!")
                return True
        except Exception as e:
            print("   [ERROR] Alternative method also failed: {}".format(str(e)))

        return False

def compare_old_new():
    """
    Compare the old and new library formats
    """
    print("\n" + "="*70)
    print("COMPARING OLD VS NEW FORMAT")
    print("="*70)

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'amd_mapping', 'data', 'outputs')

    old_file = os.path.join(data_dir, 'endmember_library.sli.hdr')
    new_file = os.path.join(data_dir, 'endmember_library_fixed.hdr')

    if not os.path.exists(old_file):
        print("[WARNING] Old file not found")
        return

    if not os.path.exists(new_file):
        print("[WARNING] New file not found")
        return

    try:
        old_lib = envi.open(old_file)
        new_lib = envi.open(new_file)

        print("\nOLD FORMAT:")
        print("  File Type: {}".format(old_lib.metadata.get('file type')))
        print("  Dimensions: {} samples x {} lines x {} bands".format(
            old_lib.metadata['samples'], old_lib.metadata['lines'], old_lib.metadata['bands']))
        print("  Interleave: {}".format(old_lib.metadata.get('interleave')))
        print("  Data Type: {}".format(old_lib.metadata.get('data type')))
        print("  Wavelength Units: {}".format(old_lib.metadata.get('wavelength units', 'MISSING')))

        print("\nNEW FORMAT (CORRECTED):")
        print("  File Type: {}".format(new_lib.metadata.get('file type')))
        print("  Dimensions: {} samples x {} lines x {} bands".format(
            new_lib.metadata['samples'], new_lib.metadata['lines'], new_lib.metadata['bands']))
        print("  Interleave: {}".format(new_lib.metadata.get('interleave')))
        print("  Data Type: {}".format(new_lib.metadata.get('data type')))
        print("  Wavelength Units: {}".format(new_lib.metadata.get('wavelength units', 'MISSING')))

        # Load and compare data
        print("\nDATA COMPARISON:")
        old_data = old_lib.load()
        new_data = new_lib.load()

        print("  Old data shape: {}".format(old_data.shape))
        print("  New data shape: {}".format(new_data.shape))

        # Extract first spectrum from each and compare
        if len(old_data.shape) == 3:
            old_spec0 = old_data[:, 0, 0]
        else:
            old_spec0 = old_data[:, 0]

        new_spec0 = new_data[0, :, 0]

        print("\n  First spectrum comparison:")
        print("    Old: min={:.4f}, max={:.4f}, mean={:.4f}".format(
            np.min(old_spec0), np.max(old_spec0), np.mean(old_spec0)))
        print("    New: min={:.4f}, max={:.4f}, mean={:.4f}".format(
            np.min(new_spec0), np.max(new_spec0), np.mean(new_spec0)))

        # Check if values match
        if np.allclose(old_spec0, new_spec0):
            print("    [OK] Spectral values match!")
        else:
            print("    [WARNING] Spectral values differ")

    except Exception as e:
        print("[ERROR] Comparison failed: {}".format(str(e)))

if __name__ == "__main__":
    print("\n")
    success = rebuild_endmember_library()

    if success:
        print("\n" + "="*70)
        print("REBUILD COMPLETE - SUMMARY")
        print("="*70)
        print("\nYour new SNAP-compatible library files:")
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, 'amd_mapping', 'data', 'outputs')
        print("  - {}/endmember_library_fixed.hdr".format(data_dir))
        print("  - {}/endmember_library_fixed".format(data_dir))

        print("\nTo use in SNAP:")
        print("  1. Open the .hdr file in SNAP")
        print("  2. It should now load without errors")
        print("  3. You can visualize the spectra in SNAP's Spectrum View")

        # Run comparison
        compare_old_new()
    else:
        print("\n[FAILED] Rebuild unsuccessful - check errors above")
