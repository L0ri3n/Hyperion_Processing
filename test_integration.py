"""
Test script to verify USGS loader integration with hyperion_workflow
"""

import sys
import os

# Test the import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'amd_mapping', 'code'))

print("Testing USGS loader integration...")
print("-" * 50)

# Test 1: Import the module
try:
    from load_usgs_spectrum import load_minerals_spectra
    print("[OK] Successfully imported load_minerals_spectra")
except ImportError as e:
    print("[ERROR] Failed to import: {}".format(e))
    sys.exit(1)

# Test 2: Load minerals
print("\nLoading mineral spectra...")
spectra = load_minerals_spectra()

if spectra:
    print("\n[OK] Successfully loaded {} minerals:".format(len(spectra)))
    for mineral_name, (wavelengths, reflectance) in spectra.items():
        print("  - {}: {} bands, wavelength range {:.1f}-{:.1f} nm".format(
            mineral_name, len(wavelengths), wavelengths.min(), wavelengths.max()))
else:
    print("[ERROR] No spectra loaded")

# Test 3: Test integration with create_endmember_library
print("\n" + "-" * 50)
print("Testing create_endmember_library function...")

# Import the workflow functions
try:
    from hyperion_workflow import create_endmember_library, USGS_LOADER_AVAILABLE
    print("[OK] Successfully imported hyperion_workflow functions")
    print("USGS_LOADER_AVAILABLE: {}".format(USGS_LOADER_AVAILABLE))
except ImportError as e:
    print("[ERROR] Failed to import hyperion_workflow: {}".format(e))
    sys.exit(1)

# Test creating endmember library
print("\nCreating endmember library...")
try:
    endmembers, wavelengths = create_endmember_library(
        output_file='./amd_mapping/data/outputs/test_endmember_library.sli'
    )

    if endmembers:
        print("\n[OK] Endmember library created successfully!")
        print("Number of endmembers: {}".format(len(endmembers)))
        print("Number of bands: {}".format(len(wavelengths)))
        print("Wavelength range: {:.1f} - {:.1f} nm".format(wavelengths.min(), wavelengths.max()))

        print("\nEndmember names:")
        for name in endmembers.keys():
            print("  - {}".format(name))
    else:
        print("[ERROR] No endmembers created")

except Exception as e:
    print("[ERROR] Failed to create endmember library: {}".format(e))
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("Integration test complete!")
