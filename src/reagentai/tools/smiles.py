import heapq
import logging

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from src.reagentai.tools.popular_smiles_dataset import SMILES_DEFAULT_LIST

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

    logger.info(f"SMILES: {smiles}, Valid: {mol is not None}")

    return mol is not None


def find_similar_molecules(
        query_smiles: str, target_smiles_list: list[str] = SMILES_DEFAULT_LIST, top_n: int = 5
) -> list[tuple[str, float]]:
    """
    Finds molecules similar to a query SMILES string from a list of target SMILES strings
    based on Tanimoto similarity of Morgan fingerprints.

    This function computes molecular similarity using RDKit's Morgan fingerprints (ECFP4-like
    circular fingerprints with radius 2) and Tanimoto similarity coefficient. It's useful for
    finding structurally similar compounds to a query molecule.

    Args:
        query_smiles (str): The SMILES string of the query molecule. Must be a valid SMILES
                           string that RDKit can parse.
        target_smiles_list (list[str], optional): A list of SMILES strings of molecules to
                                                 compare against. Defaults to SMILES_DEFAULT_LIST,
                                                 which contains a curated set of ~16,000 drug-like
                                                 molecules commonly used in chemical informatics.
        top_n (int, optional): The number of most similar molecules to return. Defaults to 5.

    Returns:
        list[tuple[str, float]]: A list of tuples, where each tuple contains
                                 the SMILES string of a similar molecule and its
                                 Tanimoto similarity score to the query molecule.
                                 The list is sorted by similarity in descending order
                                 (highest similarity first). Similarity scores range
                                 from 0 (completely dissimilar) to 1 (identical).

    Raises:
        ValueError: If the query_smiles is invalid or cannot be parsed by RDKit.

    Example:
        >>> query = "CCO"  # Ethanol
        >>> similar_mols = find_similar_molecules(query, top_n=3)
        >>> # Returns something like [("CCCO", 0.85), ("CCN", 0.78), ("CC(=O)O", 0.65)]
    """
    logger.info(
        f"[TASK] [FIND_SIMILAR_MOLECULES] Arguments: query_smiles: {query_smiles}, "
        f"number of targets: {len(target_smiles_list)}, top_n: {top_n}"
    )

    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        logger.error(f"Invalid query SMILES string: {query_smiles}")
        raise ValueError(f"Invalid query SMILES string: {query_smiles}")

    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)

    similarities = []
    for target_smiles in target_smiles_list:
        if not target_smiles or query_smiles == target_smiles:  # Skip empty or identical SMILES
            continue
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            logger.warning(f"Skipping invalid target SMILES: {target_smiles}")
            continue

        target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
        similarity = DataStructs.TanimotoSimilarity(query_fp, target_fp)
        similarities.append((target_smiles, similarity))

    # Use heapq.nlargest for efficient selection of top N similar molecules
    result = heapq.nlargest(top_n, similarities, key=lambda x: x[1])

    logger.info(f"Found {len(similarities)} similar molecules for {query_smiles}.")
    logger.debug(f"Output find_similar_molecules: {result}")
    return result
