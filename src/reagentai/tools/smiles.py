import logging
import tempfile

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

logger = logging.getLogger(__name__)


def is_valid_smiles(smiles: str, sanitize: bool = True) -> bool:
    """
    Check if a SMILES string is valid.

    Args:
        smiles (str): The SMILES string to check.
        sanitize (bool): Whether to sanitize the molecule. Default is True.

    Returns:
        bool: True if the SMILES string is valid, False otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    except Exception:
        mol = None

    logging.info(f"SMILES: {smiles}, Valid: {mol is not None}")

    return mol is not None


def image_from_smiles(smiles: str, size: tuple[int, int] = (600, 300)) -> str:
    """
    Generate an image from a SMILES string.

    Args:
        smiles (str): The SMILES string to convert to an image.
        size (tuple[int, int]): The size of the image in pixels. Default is (600, 300).

    Returns:
        str: The file path to the generated image.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    PIL_img: Image.Image = Draw.MolToImage(mol, size=size, kekulize=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        PIL_img.save(tmp_file, format="PNG")
        temp_file_path = tmp_file.name

    return temp_file_path
