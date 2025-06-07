import logging
import tempfile

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from src.reagentai.models.retrosynthesis import Route

from .helpers import RouteImageFactory

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


def image_from_smiles(smiles: str, size: tuple[int, int] = (300, 300)) -> str:
    """
    Generate an image from a SMILES string.

    Args:
        smiles (str): The SMILES string to convert to an image.
        size (tuple[int, int]): The size of the image in pixels. Default is (300, 300).

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


def route_to_image(routes: Route) -> str:
    """
    Generate an image from a SMILES route.

    Args:
        routes (RouteCollection): The collection of retrosynthesis routes.
        idx (int): The index of the route to generate an image for. Default is 0.

    Returns:
        str: The file path to the generated image.
    """
    print("Generating image for route...")
    image = RouteImageFactory(routes).image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        image.save(tmp_file, format="PNG")
        temp_file_path = tmp_file.name

    return temp_file_path
