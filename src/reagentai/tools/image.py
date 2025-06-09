import logging
import tempfile

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

from src.reagentai.common.utils.image import RouteImageFactory
from src.reagentai.models.output import ImageOutput
from src.reagentai.models.retrosynthesis import Route

logger = logging.getLogger(__name__)


def smiles_to_image(
    smiles: str, title: str, size: tuple[int, int] = (600, 300)
) -> ImageOutput:
    """
    Generate an image from a SMILES string.

    Args:
        smiles (str): The SMILES string to convert to an image.
        title (str): A title for the generated image.
        size (tuple[int, int]): The size of the image in pixels. Default is (600, 300).

    Returns:
        ImageOutput: An object containing the file path to the generated image and its description.
    Raises:
        ValueError: If the provided SMILES string is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    PIL_img: Image.Image = Draw.MolToImage(mol, size=size, kekulize=True)

    with tempfile.NamedTemporaryFile(
        prefix="reagentai_smiles_", suffix=".png", delete=False
    ) as tmp_file:
        PIL_img.save(tmp_file, format="PNG")
        temp_file_path = tmp_file.name

    logger.info(f"Generated image for SMILES: {smiles}, saved to {temp_file_path}")
    return ImageOutput(file_path=temp_file_path, title=title)


def route_to_image(route: Route, title: str) -> ImageOutput:
    """
    Generate an image from a retrosynthesis route.

    Args:
        route (Route): The retrosynthesis route to convert to an image.
        title (str): A title for the generated image.

    Returns:
        ImageOutput: An object containing the file path to the generated image and its description.
    """
    image = RouteImageFactory(route).image

    with tempfile.NamedTemporaryFile(
        prefix="reagentai_route_", suffix=".png", delete=False
    ) as tmp_file:
        image.save(tmp_file, format="PNG")
        temp_file_path = tmp_file.name

    logger.info(f"Generated image for route, saved to {temp_file_path}")
    return ImageOutput(file_path=temp_file_path, title=title)
