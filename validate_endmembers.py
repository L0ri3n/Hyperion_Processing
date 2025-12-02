"""
Test script to validate existing endmember library files
and check SNAP compatibility
"""

import numpy as np
import os
import sys
from spectral import envi

# Add the workflow functions
sys.path.insert(0, os.path.dirname(__file__))

def check_file_exists(filepath):
    """Check if file exists and print info"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print("  [FOUND] {} ({:.2f} KB)".format(filepath, size / 1024))
        return True
    else:
        print("  [MISSING] {}".format(filepath))
        return False

def validate_envi_library_detailed(library_path):
    """
    Detailed validation of ENVI spectral library file

    Parameters:
    -----------
    library_path : str
        Path to .hdr file

    Returns:
    --------
    bool : True if valid, False otherwise
    """
    print("\n" + "="*70)
    print("VALIDATING: {}".format(library_path))
    print("="*70)

    # Check if files exist
    if library_path.endswith('.hdr'):
        hdr_file = library_path
        data_file = library_path[:-4]
    else:
        hdr_file = library_path + '.hdr'
        data_file = library_path

    print("\n1. File existence check:")
    hdr_exists = check_file_exists(hdr_file)
    data_exists = check_file_exists(data_file)

    if not (hdr_exists and data_exists):
        print("\n[FAILED] Missing files")
        return False

    # Try to open and read metadata
    print("\n2. Opening ENVI library...")
    try:
        lib = envi.open(hdr_file)
        print("  [OK] Library opened successfully")
    except Exception as e:
        print("  [ERROR] Failed to open: {}".format(str(e)))
        return False

    # Check metadata
    print("\n3. Metadata check:")
    metadata_fields = {
        'file type': 'File Type',
        'samples': 'Samples (wavelength points)',
        'lines': 'Lines',
        'bands': 'Bands (spectra)',
        'data type': 'Data Type',
        'interleave': 'Interleave',
        'byte order': 'Byte Order',
        'wavelength': 'Wavelengths',
        'wavelength units': 'Wavelength Units',
        'spectra names': 'Spectra Names'
    }

    all_fields_ok = True
    for field, description in metadata_fields.items():
        if field in lib.metadata:
            value = lib.metadata[field]
            if isinstance(value, list) and len(value) > 5:
                print("  [OK] {}: [list with {} elements]".format(description, len(value)))
            else:
                print("  [OK] {}: {}".format(description, value))
        else:
            print("  [MISSING] {}".format(description))
            all_fields_ok = False

    if not all_fields_ok:
        print("\n[WARNING] Some metadata fields are missing")

    # Validate file type
    print("\n4. File type validation:")
    if lib.metadata.get('file type') == 'ENVI Spectral Library':
        print("  [OK] Correct file type")
    else:
        print("  [ERROR] Wrong file type: {}".format(lib.metadata.get('file type')))
        return False

    # Validate dimensions
    print("\n5. Dimension validation:")
    n_spectra = lib.metadata.get('bands', 0)
    n_bands = lib.metadata.get('samples', 0)
    n_lines = lib.metadata.get('lines', 0)

    print("  Lines: {} (should be 1)".format(n_lines))
    print("  Samples: {} (spectral bands per spectrum)".format(n_bands))
    print("  Bands: {} (number of spectra)".format(n_spectra))

    if n_lines != 1:
        print("  [WARNING] Lines should be 1 for spectral library")

    # Validate interleave
    print("\n6. Interleave format:")
    interleave = lib.metadata.get('interleave', 'unknown')
    print("  Interleave: {}".format(interleave))
    if interleave.lower() == 'bil':
        print("  [OK] BIL is standard for spectral libraries")
    else:
        print("  [WARNING] BIL is recommended for spectral libraries")

    # Try to load data
    print("\n7. Data loading test:")
    try:
        data = lib.load()
        print("  [OK] Data loaded successfully")
        print("  Data shape: {}".format(data.shape))
        print("  Data type: {}".format(data.dtype))

        # Check for invalid values
        print("\n8. Data quality check:")
        n_nan = np.sum(np.isnan(data))
        n_inf = np.sum(np.isinf(data))
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        data_mean = np.nanmean(data)

        print("  Min value: {:.6f}".format(data_min))
        print("  Max value: {:.6f}".format(data_max))
        print("  Mean value: {:.6f}".format(data_mean))
        print("  NaN values: {}".format(n_nan))
        print("  Inf values: {}".format(n_inf))

        if n_nan > 0 or n_inf > 0:
            print("  [ERROR] Data contains invalid values")
            return False

        # Check value range
        if data_min < 0 or data_max > 2.0:
            print("  [WARNING] Reflectance values outside expected range [0, 1.5]")

        # Extract and display spectra info
        print("\n9. Spectra information:")
        spectra_names = lib.metadata.get('spectra names', [])
        print("  Number of spectra: {}".format(len(spectra_names)))
        for i, name in enumerate(spectra_names):
            # Extract spectrum (depends on data shape)
            if len(data.shape) == 3:
                spectrum = data[0, :, i]
            elif len(data.shape) == 2:
                spectrum = data[:, i]
            else:
                spectrum = data.flatten()

            spec_min = np.min(spectrum)
            spec_max = np.max(spectrum)
            spec_mean = np.mean(spectrum)
            print("  {}: min={:.4f}, max={:.4f}, mean={:.4f}".format(
                name, spec_min, spec_max, spec_mean))

        print("\n" + "="*70)
        print("[SUCCESS] Library validation PASSED")
        print("="*70)
        return True

    except Exception as e:
        print("  [ERROR] Failed to load data: {}".format(str(e)))
        print("\n" + "="*70)
        print("[FAILED] Library validation FAILED")
        print("="*70)
        return False

def main():
    """Main validation routine"""
    print("\n" + "="*70)
    print("ENDMEMBER LIBRARY VALIDATION TOOL")
    print("="*70)

    # Define paths to check
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'amd_mapping', 'data', 'outputs')

    # Possible endmember library files
    library_files = [
        os.path.join(data_dir, 'endmember_library.sli'),
        os.path.join(data_dir, 'endmember_library_matched.sli'),
        os.path.join(data_dir, 'endmember_library'),
        os.path.join(data_dir, 'endmember_library_matched'),
    ]

    # Also check for .hdr files directly
    for filename in ['endmember_library.hdr', 'endmember_library_matched.hdr']:
        library_files.append(os.path.join(data_dir, filename))

    print("\nSearching for endmember libraries in:")
    print("  {}".format(data_dir))

    # Find existing files
    found_files = []
    for lib_file in library_files:
        # Check both with and without .hdr
        for path in [lib_file, lib_file + '.hdr']:
            if os.path.exists(path) and path not in found_files:
                found_files.append(path)

    if not found_files:
        print("\n[ERROR] No endmember library files found!")
        print("\nSearched locations:")
        for lib_file in library_files:
            print("  - {}".format(lib_file))
        return

    print("\nFound {} library file(s):".format(len(found_files)))
    for f in found_files:
        print("  - {}".format(f))

    # Validate each found file
    results = {}
    for lib_file in found_files:
        if lib_file.endswith('.hdr') or not lib_file.endswith('.csv'):
            is_valid = validate_envi_library_detailed(lib_file)
            results[lib_file] = is_valid

    # Summary
    print("\n\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    for lib_file, is_valid in results.items():
        status = "PASSED" if is_valid else "FAILED"
        print("{}: {}".format(os.path.basename(lib_file), status))

    if all(results.values()):
        print("\n[SUCCESS] All libraries are SNAP-compatible!")
    else:
        print("\n[WARNING] Some libraries have issues - see details above")

if __name__ == "__main__":
    main()
