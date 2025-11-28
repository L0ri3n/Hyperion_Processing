# load_usgs_spectrum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIGURATION: Path to minerals folder
MINERALS_DIR = r"C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\PROCESSING_AND_POST\amd_mapping\data\spectral_library\usgs\ASCIIdata\ASCIIdata_splib07b_cvHYPERION\ChapterM_Minerals"

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

def load_usgs_spectrum(filepath):
    """
    Load USGS spectral library file

    USGS format (after header):
    Column 1: Wavelength (micrometers or nanometers)
    Column 2: Reflectance
    """
    # Read the file, skip header lines
    data = pd.read_csv(filepath, sep='\s+', skiprows=18, header=None)

    wavelength = data.iloc[:, 0].values
    reflectance = data.iloc[:, 1].values

    # Convert to nm if in micrometers (check if max < 10)
    if wavelength.max() < 10:
        wavelength = wavelength * 1000  # Convert um to nm

    return wavelength, reflectance

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

    spectra = {}

    for mineral in mineral_list:
        filepath = find_mineral_file(mineral, minerals_dir)

        if filepath:
            try:
                wvl, ref = load_usgs_spectrum(filepath)
                spectra[mineral] = (wvl, ref)
                print(f"[OK] Loaded: {mineral} ({len(wvl)} points)")
            except Exception as e:
                print(f"[ERROR] Error loading {mineral}: {e}")
        else:
            print(f"[NOT FOUND] File not found for: {mineral}")

    return spectra

# Example usage
if __name__ == "__main__":
    # Load minerals specified in MINERALS_TO_LOAD
    spectra = load_minerals_spectra()

    print(f"\nTotal spectra loaded: {len(spectra)}")

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
        plt.savefig('outputs/figures/minerals_spectra.png', dpi=150, bbox_inches='tight')
        plt.show()
