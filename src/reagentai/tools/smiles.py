import logging

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

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
    query_smiles: str, target_smiles_list: list[str], top_n: int = 5
) -> list[tuple[str, float]]:
    """
    Finds molecules similar to a query SMILES string from a list of target SMILES strings
    based on Tanimoto similarity of Morgan fingerprints.

    Args:
        query_smiles (str): The SMILES string of the query molecule.
        target_smiles_list (list[str]): A list of SMILES strings of molecules to compare against.
        top_n (int): The number of most similar molecules to return.

    Returns:
        list[tuple[str, float]]: A list of tuples, where each tuple contains
                                 the SMILES string of a similar molecule and its
                                 Tanimoto similarity score to the query molecule.
                                 The list is sorted by similarity in descending order.

    Raises:
        ValueError: If the query_smiles is invalid.
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

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"Found {len(similarities)} similar molecules for {query_smiles}.")

    result = similarities[:top_n]
    logger.debug(f"Output find_similar_molecules: {result}")
    return result
