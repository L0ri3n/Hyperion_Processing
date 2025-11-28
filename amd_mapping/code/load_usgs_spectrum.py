# load_usgs_spectrum
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIGURATION: Path to minerals folder
MINERALS_DIR = r"C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\usgs\ASCIIdata\ASCIIdata_splib07b_cvHYPERION\ChapterM_Minerals"

# CONFIGURATION: List of minerals to load
# Add or remove mineral names as needed
MINERALS_TO_LOAD = [
    'Jarosite',
    'Goethite',
    'Hematite',
    'Kaolinite',
    'Alunite',
    'Illite',
    'Smectite',
    'Gypsum',
]

def load_hyperion_wavelengths(wavelength_file):
    """
    Load Hyperion wavelengths from the wavelength file

    Returns:
        Wavelength array in nanometers
    """
    # Skip first line (header), then read all wavelength values
    wavelengths = []
    with open(wavelength_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            wavelengths.append(float(line.strip()))

    wavelengths = np.array(wavelengths)

    # Convert micrometers to nanometers
    if wavelengths.max() < 10:
        wavelengths = wavelengths * 1000

    return wavelengths

def load_usgs_spectrum(filepath, wavelengths):
    """
    Load USGS spectral library file (Hyperion format)

    USGS Hyperion format:
    - First line: header
    - Remaining lines: reflectance values only (one per line)

    Args:
        filepath: Path to the spectrum file
        wavelengths: Wavelength array (loaded separately)

    Returns:
        wavelength, reflectance arrays
    """
    # Read reflectance values, skip first line (header)
    reflectance = []
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            reflectance.append(float(line.strip()))

    reflectance = np.array(reflectance)

    return wavelengths, reflectance

def find_mineral_file(mineral_name, minerals_dir):
    """
    Search for the file corresponding to the mineral in the specified folder

    Args:
        mineral_name: Name of the mineral to search for
        minerals_dir: Directory to search in

    Returns:
        Full path to the file or None if not found
    """
    minerals_path = Path(minerals_dir)

    # Search for files containing the mineral name (case-insensitive)
    for file in minerals_path.glob('*.txt'):
        if mineral_name.lower() in file.stem.lower():
            return str(file)

    return None

def load_minerals_spectra(mineral_list=None, minerals_dir=MINERALS_DIR):
    """
    Load spectra for a list of minerals

    Args:
        mineral_list: List of mineral names. If None, uses MINERALS_TO_LOAD
        minerals_dir: Directory to search for files

    Returns:
        dict: Dictionary with {mineral_name: (wavelength, reflectance)}
    """
    if mineral_list is None:
        mineral_list = MINERALS_TO_LOAD

    # Load wavelengths from the Hyperion wavelength file
    wavelength_file = Path(minerals_dir).parent / "s07HYPRN_Hyperion_Wavelengths_microns_AVG_xtrack.txt"

    try:
        wavelengths = load_hyperion_wavelengths(str(wavelength_file))
        print("Loaded {} wavelength points from {} to {} nm".format(
            len(wavelengths), wavelengths.min(), wavelengths.max()))
    except Exception as e:
        print("[ERROR] Failed to load wavelength file: {}".format(e))
        return {}

    spectra = {}

    for mineral in mineral_list:
        filepath = find_mineral_file(mineral, minerals_dir)

        if filepath:
            try:
                wvl, ref = load_usgs_spectrum(filepath, wavelengths)
                spectra[mineral] = (wvl, ref)
                print("[OK] Loaded: {} ({} points)".format(mineral, len(wvl)))
            except Exception as e:
                print("[ERROR] Error loading {}: {}".format(mineral, e))
        else:
            print("[NOT FOUND] File not found for: {}".format(mineral))

    return spectra

# Example usage
if __name__ == "__main__":
    # Load minerals specified in MINERALS_TO_LOAD
    spectra = load_minerals_spectra()

    print("\nTotal spectra loaded: {}".format(len(spectra)))

    # Plot all loaded spectra
    if spectra:
        plt.figure(figsize=(12, 6))

        for mineral, (wvl, ref) in spectra.items():
            plt.plot(wvl, ref, label=mineral, alpha=0.7)

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title('Mineral Spectra from USGS Library')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Create output directory in amd_mapping/data
        import os
        output_dir = '../data/outputs'
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, 'minerals_spectra.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print("\nPlot saved to: {}".format(output_file))
        plt.show()
