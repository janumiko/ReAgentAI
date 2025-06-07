import logging
import tempfile

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from src.reagentai.models.retrosynthesis import Route

from .helpers import RouteImageFactory

logger = logging.getLogger(__name__)


def smiles_to_image(smiles: str, size: tuple[int, int] = (600, 300)) -> str:
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

    with tempfile.NamedTemporaryFile(prefix="reagentai_smiles_", suffix=".png", delete=False) as tmp_file:
        PIL_img.save(tmp_file, format="PNG")
        temp_file_path = tmp_file.name

    logger.info(f"Generated image for SMILES: {smiles}, saved to {temp_file_path}")
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
    image = RouteImageFactory(routes).image

    with tempfile.NamedTemporaryFile(prefix="reagentai_route_", suffix=".png", delete=False) as tmp_file:
        image.save(tmp_file, format="PNG")
        temp_file_path = tmp_file.name

    logger.info(f"Generated image for route, saved to {temp_file_path}")
    return temp_file_path
