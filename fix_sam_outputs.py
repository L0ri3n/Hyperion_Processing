"""
Fix SAM classification output files for SNAP compatibility
The issue: envi.save_image creates .img files, but headers point to files without extension
"""

import os
import shutil

def fix_envi_file_extensions(directory):
    """
    Rename .img files to match what headers expect (no extension)
    or update headers to point to .img files
    """
    print("="*70)
    print("FIXING SAM OUTPUT FILES FOR SNAP COMPATIBILITY")
    print("="*70)

    # Find all .hdr files
    hdr_files = [f for f in os.listdir(directory) if f.endswith('.hdr')]

    print("\nFound {} header files".format(len(hdr_files)))

    for hdr_file in hdr_files:
        base_name = hdr_file[:-4]  # Remove .hdr
        hdr_path = os.path.join(directory, hdr_file)

        # Expected data files (in order of preference for SNAP)
        data_file_noext = os.path.join(directory, base_name)
        data_file_img = os.path.join(directory, base_name + '.img')
        data_file_dat = os.path.join(directory, base_name + '.dat')

        print("\nProcessing: {}".format(hdr_file))

        # Check which data file exists
        if os.path.exists(data_file_noext):
            print("  [OK] Data file exists (no extension): {}".format(base_name))
            continue

        elif os.path.exists(data_file_img):
            print("  [FOUND] Data file with .img extension")
            print("  [ACTION] Copying .img to no-extension file for SNAP compatibility")

            try:
                # Copy (don't move, in case .img is needed by other tools)
                shutil.copy2(data_file_img, data_file_noext)
                size_mb = os.path.getsize(data_file_noext) / (1024**2)
                print("  [SUCCESS] Created: {} ({:.2f} MB)".format(base_name, size_mb))
            except Exception as e:
                print("  [ERROR] Failed to copy: {}".format(str(e)))

        elif os.path.exists(data_file_dat):
            print("  [FOUND] Data file with .dat extension")
            print("  [ACTION] Copying .dat to no-extension file")

            try:
                shutil.copy2(data_file_dat, data_file_noext)
                print("  [SUCCESS] Created: {}".format(base_name))
            except Exception as e:
                print("  [ERROR] Failed to copy: {}".format(str(e)))

        else:
            print("  [WARNING] No data file found for this header!")
            print("    Expected one of:")
            print("      - {}".format(base_name))
            print("      - {}.img".format(base_name))
            print("      - {}.dat".format(base_name))

def verify_snap_compatibility(directory):
    """
    Verify that all files are ready for SNAP
    """
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    hdr_files = [f for f in os.listdir(directory) if f.endswith('.hdr')]

    all_ok = True
    for hdr_file in hdr_files:
        base_name = hdr_file[:-4]
        hdr_path = os.path.join(directory, hdr_file)
        data_path = os.path.join(directory, base_name)

        if os.path.exists(data_path):
            size = os.path.getsize(data_path)

            # Read expected size from header
            with open(hdr_path, 'r') as f:
                content = f.read()

                # Parse dimensions
                import re
                samples = int(re.search(r'samples\s*=\s*(\d+)', content).group(1))
                lines = int(re.search(r'lines\s*=\s*(\d+)', content).group(1))
                bands = int(re.search(r'bands\s*=\s*(\d+)', content).group(1))
                data_type = int(re.search(r'data type\s*=\s*(\d+)', content).group(1))

                # Calculate expected size
                bytes_per_pixel = {1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 12: 2}[data_type]
                expected_size = samples * lines * bands * bytes_per_pixel

                if size == expected_size:
                    print("[OK] {}: {} bytes (matches header)".format(base_name, size))
                else:
                    print("[ERROR] {}: {} bytes (expected {})".format(
                        base_name, size, expected_size))
                    all_ok = False
        else:
            print("[MISSING] {}: No data file!".format(base_name))
            all_ok = False

    print("\n" + "="*70)
    if all_ok:
        print("[SUCCESS] All files are SNAP-compatible!")
    else:
        print("[WARNING] Some files have issues - see above")
    print("="*70)

    return all_ok

if __name__ == "__main__":
    import sys

    # Directory containing SAM outputs
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.path.join(os.path.dirname(__file__),
                                'amd_mapping', 'outputs', 'classifications')

    if not os.path.exists(directory):
        print("[ERROR] Directory not found: {}".format(directory))
        sys.exit(1)

    print("Working directory: {}".format(directory))

    # Fix file extensions
    fix_envi_file_extensions(directory)

    # Verify
    verify_snap_compatibility(directory)

    print("\n\nTo use in SNAP:")
    print("  1. Open SNAP")
    print("  2. File -> Import -> Optical Sensors -> Generic Formats -> ENVI")
    print("  3. Select any .hdr file from: {}".format(directory))
    print("  4. Files should now load without EOF errors")
