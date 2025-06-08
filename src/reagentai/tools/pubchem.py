import logging

import pubchempy as pcp

logger = logging.getLogger(__name__)


def name_to_smiles(chemical_name: str) -> str:
    """
    Finds the canonical SMILES string for a given chemical name using PubChem database.

    This function searches the PubChem database for a chemical compound by name and
    returns its canonical SMILES representation. PubChem is a comprehensive database
    maintained by the National Center for Biotechnology Information (NCBI) containing
    millions of chemical structures and their associated data.

    Args:
        chemical_name (str): The name of the chemical compound to search for.
                           Can be a common name (e.g., "aspirin", "caffeine"),
                           IUPAC name, or other chemical identifier.

    Returns:
        str: The canonical SMILES string of the compound as found in PubChem.

    Raises:
        ValueError: If no compound is found for the given name or if multiple
                   ambiguous results are returned without a clear match.

    Example:
        >>> smiles = name_to_smiles("aspirin")
        >>> # Returns "CC(=O)OC1=CC=CC=C1C(=O)O"
        >>> smiles = name_to_smiles("caffeine")
        >>> # Returns "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    """
    logger.info(f"[TASK] [NAME_TO_SMILES] Arguments: chemical_name: {chemical_name}")

    if not chemical_name or not chemical_name.strip():
        logger.error("Empty or invalid chemical name provided")
        raise ValueError("Chemical name cannot be empty")

    chemical_name = chemical_name.strip()

    try:
        # Search for the compound by name
        compounds = pcp.get_compounds(chemical_name, "name")

        if not compounds:
            logger.warning(f"No compounds found for name: {chemical_name}")
            raise ValueError(f"No compound found for name: {chemical_name}")

        # Get the first (most relevant) compound
        compound = compounds[0]

        # Get canonical SMILES
        smiles = compound.canonical_smiles

        if not smiles:
            logger.error(f"No SMILES found for compound: {chemical_name}")
            raise ValueError(f"No SMILES representation found for: {chemical_name}")

        logger.info(f"Found SMILES for '{chemical_name}': {smiles}")
        logger.debug(f"PubChem CID: {compound.cid}")

        return smiles

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        logger.error(f"Error searching PubChem for '{chemical_name}': {str(e)}")
        raise ValueError(f"Failed to retrieve SMILES for '{chemical_name}': {str(e)}") from e
