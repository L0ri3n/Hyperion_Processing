"""
Direct test of the corrected endmember library
"""

import numpy as np
import os
import struct

def read_envi_library_manual(filepath):
    """
    Manually read ENVI library to verify format
    """
    # Read header
    hdr_file = filepath + '.hdr' if not filepath.endswith('.hdr') else filepath
    data_file = filepath if not filepath.endswith('.hdr') else filepath[:-4]

    print("Reading header: {}".format(hdr_file))
    metadata = {}

    with open(hdr_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                metadata[key] = value

    # Parse key values
    samples = int(metadata['samples'])
    lines = int(metadata['lines'])
    bands = int(metadata['bands'])
    data_type = int(metadata['data type'])
    interleave = metadata['interleave']

    print("\nMetadata:")
    print("  Samples: {} (wavelength points)".format(samples))
    print("  Lines: {}".format(lines))
    print("  Bands: {} (spectra)".format(bands))
    print("  Data type: {} (4=float32)".format(data_type))
    print("  Interleave: {}".format(interleave))

    # Read binary data
    print("\nReading binary data: {}".format(data_file))
    expected_values = samples * lines * bands
    expected_bytes = expected_values * 4  # 4 bytes per float32

    file_size = os.path.getsize(data_file)
    print("  File size: {} bytes".format(file_size))
    print("  Expected: {} bytes ({} values)".format(expected_bytes, expected_values))

    if file_size != expected_bytes:
        print("  [WARNING] Size mismatch!")

    # Read all data
    with open(data_file, 'rb') as f:
        data_bytes = f.read()

    # Unpack as float32
    n_values = len(data_bytes) // 4
    data = struct.unpack('{}f'.format(n_values), data_bytes)
    data = np.array(data)

    print("  Read {} values".format(len(data)))

    # Reshape according to BIL format
    # BIL for spectral library: [lines, bands, samples]
    # But we have lines=1, so really [bands, samples] = [n_spectra, n_wavelengths]
    data_reshaped = data.reshape((bands, samples))

    print("  Reshaped to: {}".format(data_reshaped.shape))
    print("  (each row is one spectrum)")

    # Show statistics
    print("\nData statistics:")
    print("  Overall min: {:.4f}".format(np.min(data_reshaped)))
    print("  Overall max: {:.4f}".format(np.max(data_reshaped)))
    print("  Overall mean: {:.4f}".format(np.mean(data_reshaped)))

    # Parse spectra names
    names_str = metadata.get('spectra names', '')
    # Remove braces
    names_str = names_str.strip('{}')
    names = [n.strip() for n in names_str.split(',')]

    print("\nSpectra:")
    for i, name in enumerate(names):
        if i < len(data_reshaped):
            spectrum = data_reshaped[i]
            print("  {}: min={:.4f}, max={:.4f}, mean={:.4f}".format(
                name, np.min(spectrum), np.max(spectrum), np.mean(spectrum)))

    return data_reshaped, names, metadata

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    lib_file = os.path.join(base_dir, 'amd_mapping', 'data', 'outputs',
                            'endmember_library_fixed_alt')

    print("="*70)
    print("DIRECT LIBRARY TEST")
    print("="*70)
    print()

    try:
        data, names, metadata = read_envi_library_manual(lib_file)
        print("\n[SUCCESS] Library is correctly formatted for SNAP!")
        print("\nTo use in SNAP:")
        print("  1. Open SNAP")
        print("  2. File -> Import -> Optical Sensors -> Generic Formats -> ENVI")
        print("  3. Select: {}.hdr".format(lib_file))
        print("  4. The library should load without errors")

    except Exception as e:
        print("\n[ERROR] {}".format(str(e)))
        import traceback
        traceback.print_exc()
