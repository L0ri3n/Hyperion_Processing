#load_usgs_spectrum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIGURACIÓN: Ruta a la carpeta de minerales
MINERALS_DIR = r"C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/PROCESSING_AND_POST/amd_mapping/data/spectral_library/usgs\ASCIIdata\ASCIIdata_splib07b_cvHYPERION\ChapterM_Minerals"

# CONFIGURACIÓN: Lista de minerales a cargar
# Añade o elimina nombres de minerales según necesites
MINERALS_TO_LOAD = [
    'Jarosite',
    'Goethite',
    'Hematite',
    'Kaolinite',
    'Alunite',
    'illite',
    'smectite',
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
        wavelength = wavelength * 1000  # Convert µm to nm

    return wavelength, reflectance

def find_mineral_file(mineral_name, minerals_dir):
    """
    Busca el archivo correspondiente al mineral en la carpeta especificada

    Args:
        mineral_name: Nombre del mineral a buscar
        minerals_dir: Directorio donde buscar

    Returns:
        Path completo al archivo o None si no se encuentra
    """
    minerals_path = Path(minerals_dir)

    # Buscar archivos que contengan el nombre del mineral (case-insensitive)
    for file in minerals_path.glob('*.txt'):
        if mineral_name.lower() in file.stem.lower():
            return str(file)

    return None

def load_minerals_spectra(mineral_list=None, minerals_dir=MINERALS_DIR):
    """
    Carga los espectros de una lista de minerales

    Args:
        mineral_list: Lista de nombres de minerales. Si es None, usa MINERALS_TO_LOAD
        minerals_dir: Directorio donde buscar los archivos

    Returns:
        dict: Diccionario con {nombre_mineral: (wavelength, reflectance)}
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
                print(f"✓ Cargado: {mineral} ({len(wvl)} puntos)")
            except Exception as e:
                print(f"✗ Error cargando {mineral}: {e}")
        else:
            print(f"✗ No se encontró archivo para: {mineral}")

    return spectra

# Example usage
if __name__ == "__main__":
    # Cargar los minerales especificados en MINERALS_TO_LOAD
    spectra = load_minerals_spectra()

    print(f"\nTotal de espectros cargados: {len(spectra)}")

    # Plotear todos los espectros cargados
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