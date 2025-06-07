from typing import Literal

from src.reagentai.models.retrosynthesis import (
    MolNode,
    ReactionMetadata,
    ReactionNode,
    Route,
    ScoreData,
)


class NodeProcessingState:
    """Helper class to keep track of ID counters and collected nodes."""

    def __init__(self):
        self.mol_counter: int = 0
        self.react_counter: int = 0
        self.processed_mol_nodes: list[MolNode] = []
        self.processed_reaction_nodes: list[ReactionNode] = []

    def get_next_id(self, node_type: Literal["mol", "reaction"]) -> str:
        """
        Generates the next unique ID for a node based on its type.
        Args:
            node_type (Literal["mol", "reaction"]): The type of the node for which to generate an ID.
        Returns:
            str: A unique identifier for the node, formatted as "mol-{counter}" or "react-{counter}".
        """

        if node_type == "mol":
            self.mol_counter += 1
            return f"mol-{self.mol_counter}"
        elif node_type == "reaction":
            self.react_counter += 1
            return f"react-{self.react_counter}"
        else:
            raise ValueError(f"Unknown node type for ID generation: {node_type}")


def parse_route_score(score_dict: dict) -> ScoreData:
    """
    Parses a score dictionary from the LLM response into a ScoreData object.

    Args:
        score_dict (dict): The dictionary containing score data from the LLM response.
    Returns:
        ScoreData: An instance of ScoreData populated with the parsed values.
    """

    key_map = {
        "state score": "state_score",
        "number of reactions": "n_reactions",
        "number of pre-cursors": "n_precursors",
        "number of pre-cursors in stock": "n_precursors_in_stock",
        "average template occurrence": "avg_template_occurrence",
    }

    parsed_score_dict = {}
    for llm_key, pydantic_key in key_map.items():
        if llm_key in score_dict:
            parsed_score_dict[pydantic_key] = score_dict[llm_key]

    return ScoreData(**parsed_score_dict)


def parse_node(node_dict: dict, state: NodeProcessingState) -> str:
    """
    Recursively parses a node dictionary into a MolNode or ReactionNode object,
    updating the processing state with the parsed nodes.

    Args:
        node_dict (dict): The dictionary representation of the node.
        state (NodeProcessingState): The state object to keep track of IDs and processed nodes.
    Returns:
        str: The unique identifier for the parsed node.
    """

    node_type = node_dict["type"]
    children_data = node_dict.get("children", [])

    current_node_id = state.get_next_id(node_type)

    child_ids = []
    for child_dict in children_data:
        child_id = parse_node(child_dict, state)
        child_ids.append(child_id)

    node_data = {
        "node_id": current_node_id,
        "smiles": node_dict["smiles"],
        "children": child_ids,
    }

    if node_type == "mol":
        node_data["in_stock"] = node_dict["in_stock"]
        node_data["is_chemical"] = node_dict["is_chemical"]
        parsed_node = MolNode(**node_data)
        state.processed_mol_nodes.append(parsed_node)
    elif node_type == "reaction":
        metadata_dict = node_dict["metadata"]
        metadata = ReactionMetadata(
            classification=metadata_dict["classification"],
            policy_probability=metadata_dict["policy_probability"],
        )
        node_data["metadata"] = metadata
        parsed_node = ReactionNode(**node_data)
        state.processed_reaction_nodes.append(parsed_node)

    return current_node_id


def parse_route_dict(route_dict: dict, route_id: int) -> Route:
    """
    Parses a route dictionary into a Route object.

    Args:
        route_dict (dict): The dictionary representation of the route.
        route_id (int): The unique identifier for the route.
    Returns:
        Route: An instance of Route populated with the parsed data.
    """

    score_data = parse_route_score(route_dict["scores"])
    processing_state = NodeProcessingState()

    root_node_id = parse_node(route_dict, processing_state)

    return Route(
        route_id=route_id,
        score_data=score_data,
        root_node_id=root_node_id,
        mol_nodes=processing_state.processed_mol_nodes,
        reaction_nodes=processing_state.processed_reaction_nodes,
    )
