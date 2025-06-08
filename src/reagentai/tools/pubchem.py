import logging

import pubchempy as pcp

from src.reagentai.tools.smiles import is_valid_smiles

logger = logging.getLogger(__name__)


def get_smiles_from_name(compound_name: str) -> str:
    """
    Retrieve the SMILES string for a chemical compound using its common name via PubChem.

    This function searches the PubChem database to find the canonical SMILES representation
    of a chemical compound based on its common name, IUPAC name, or other identifiers.
    PubChem is a comprehensive chemical database maintained by the NIH that contains
    millions of chemical structures and their properties.

    Args:
        compound_name (str): The name of the chemical compound to search for.
                           This can be a common name (e.g., "aspirin", "caffeine"),
                           IUPAC name, trade name, or other chemical identifier.

    Returns:
        str: The canonical SMILES string of the compound as found in PubChem.

    Raises:
        ValueError: If the compound name is not found in PubChem or if no valid
                   SMILES string could be retrieved.
        ConnectionError: If there's a network issue connecting to PubChem servers.

    Example:
        >>> smiles = get_smiles_from_name("aspirin")
        >>> print(smiles)
        "CC(=O)OC1=CC=CC=C1C(=O)O"

        >>> smiles = get_smiles_from_name("caffeine")
        >>> print(smiles)
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    """
    logger.info(f"[TASK] [GET_SMILES_FROM_NAME] Arguments: compound_name: {compound_name}")

    if not compound_name or not compound_name.strip():
        logger.error("Empty or invalid compound name provided")
        raise ValueError("Compound name cannot be empty")

    compound_name = compound_name.strip()

    # Check if the input is already a valid SMILES string
    if is_valid_smiles(compound_name):
        logger.info(f"Input appears to be a valid SMILES, returning as is: {compound_name}")
        return compound_name

    try:
        # Search for the compound by name
        compounds = pcp.get_compounds(compound_name, "name")

        if not compounds:
            logger.warning(f"No compounds found for name: {compound_name}")
            raise ValueError(f"No compound found in PubChem for name: '{compound_name}'")

        # Get the first (most relevant) compound
        compound = compounds[0]

        # Retrieve the canonical SMILES
        smiles = compound.canonical_smiles

        if not smiles:
            logger.error(f"No SMILES found for compound: {compound_name}")
            raise ValueError(f"No SMILES string available for compound: '{compound_name}'")

        logger.info(f"Successfully retrieved SMILES for {compound_name}: {smiles}")
        logger.debug(f"PubChem CID: {compound.cid}")

        return smiles

    except Exception as e:
        if isinstance(e, ValueError):
            raise  # Re-raise ValueError as-is

        logger.error(f"Error retrieving SMILES for {compound_name}: {str(e)}")

        # Check if it's a network-related error
        if "connection" in str(e).lower() or "network" in str(e).lower():
            raise ConnectionError(f"Failed to connect to PubChem: {str(e)}") from e

        # For other exceptions, wrap in ValueError
        raise ValueError(f"Failed to retrieve SMILES for '{compound_name}': {str(e)}") from e


def get_compound_info(compound_name: str) -> dict[str, str | list | None]:
    """
    Retrieve comprehensive information about a chemical compound from PubChem.

    This function provides additional chemical information beyond just the SMILES string,
    including molecular formula, molecular weight, IUPAC name, and other identifiers.

    Args:
        compound_name (str): The name of the chemical compound to search for.

    Returns:
        dict[str, Optional[str]]: A dictionary containing compound information with keys:
            - 'smiles': Canonical SMILES string
            - 'molecular_formula': Molecular formula
            - 'molecular_weight': Molecular weight in g/mol
            - 'iupac_name': IUPAC systematic name
            - 'cid': PubChem Compound ID
            - 'synonyms': List of alternative names (first 5)

    Raises:
        ValueError: If the compound name is not found in PubChem.
        ConnectionError: If there's a network issue connecting to PubChem servers.

    Example:
        >>> info = get_compound_info("aspirin")
        >>> print(info['smiles'])
        "CC(=O)OC1=CC=CC=C1C(=O)O"
        >>> print(info['molecular_formula'])
        "C9H8O4"
    """
    logger.info(f"[TASK] [GET_COMPOUND_INFO] Arguments: compound_name: {compound_name}")

    if not compound_name or not compound_name.strip():
        logger.error("Empty or invalid compound name provided")
        raise ValueError("Compound name cannot be empty")

    compound_name = compound_name.strip()

    try:
        # Search for the compound by name
        compounds = pcp.get_compounds(compound_name, "name")

        if not compounds:
            logger.warning(f"No compounds found for name: {compound_name}")
            raise ValueError(f"No compound found in PubChem for name: '{compound_name}'")

        # Get the first (most relevant) compound
        compound = compounds[0]

        # Extract comprehensive information
        info = {
            "smiles": getattr(compound, "canonical_smiles", None),
            "molecular_formula": getattr(compound, "molecular_formula", None),
            "molecular_weight": str(getattr(compound, "molecular_weight", None))
            if hasattr(compound, "molecular_weight")
            else None,
            "iupac_name": getattr(compound, "iupac_name", None),
            "cid": str(getattr(compound, "cid", None)) if hasattr(compound, "cid") else None,
            "synonyms": getattr(compound, "synonyms", [])[:5]
            if hasattr(compound, "synonyms")
            else [],
        }

        logger.info(f"Successfully retrieved compound info for {compound_name}")
        logger.debug(f"Compound info: {info}")

        return info

    except Exception as e:
        if isinstance(e, ValueError):
            raise  # Re-raise ValueError as-is

        logger.error(f"Error retrieving compound info for {compound_name}: {str(e)}")

        # Check if it's a network-related error
        if "connection" in str(e).lower() or "network" in str(e).lower():
            raise ConnectionError(f"Failed to connect to PubChem: {str(e)}") from e

        # For other exceptions, wrap in ValueError
        raise ValueError(
            f"Failed to retrieve compound info for '{compound_name}': {str(e)}"
        ) from e


def get_name_from_smiles(smiles: str) -> str:
    """
    Retrieve the best-matching chemical name for a given SMILES string using PubChem.

    Args:
        smiles (str): The SMILES string of the compound.

    Returns:
        str: The best-matching chemical name (IUPAC or synonym) from PubChem.

    Raises:
        ValueError: If no compound is found for the SMILES or no name is available.
        ConnectionError: If there's a network issue connecting to PubChem servers.
    """
    logger.info(f"[TASK] [GET_NAME_FROM_SMILES] Arguments: smiles: {smiles}")

    if not smiles or not smiles.strip():
        logger.error("Empty or invalid SMILES provided")
        raise ValueError("SMILES string cannot be empty")

    smiles = smiles.strip()

    try:
        compounds = pcp.get_compounds(smiles, "smiles")
        if not compounds:
            logger.warning(f"No compounds found for SMILES: {smiles}")
            raise ValueError(f"No compound found in PubChem for SMILES: '{smiles}'")
        compound = compounds[0]
        # Prefer IUPAC name, fall back to first synonym
        name = getattr(compound, "iupac_name", None)
        if not name:
            synonyms = getattr(compound, "synonyms", [])
            if synonyms:
                name = synonyms[0]
        if not name:
            logger.error(f"No name found for SMILES: {smiles}")
            raise ValueError(f"No name available for SMILES: '{smiles}'")
        logger.info(f"Successfully retrieved name for SMILES {smiles}: {name}")
        return name
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        logger.error(f"Error retrieving name for SMILES {smiles}: {str(e)}")
        if "connection" in str(e).lower() or "network" in str(e).lower():
            raise ConnectionError(f"Failed to connect to PubChem: {str(e)}") from e
        raise ValueError(f"Failed to retrieve name for SMILES '{smiles}': {str(e)}") from e
