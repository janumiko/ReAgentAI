from pydantic_ai.tools import DocstringFormat, Tool

from src.reagentai.models.retrosynthesis import (
    MolNode,
    Node,
    ReactionMetadata,
    ReactionNode,
    Route,
    Score,
)


def create_tool(func, takes_ctx: bool = False) -> Tool:
    return Tool(
        function=func,
        takes_ctx=takes_ctx,
    )


def parse_node(node_dict: dict) -> Node | None:
    node_type = node_dict.get("type")
    children_data = node_dict.get("children", [])
    parsed_children = [parse_node(child) for child in children_data if isinstance(child, dict)]
    parsed_children = [child for child in parsed_children if child is not None]

    if node_type == "mol":
        return MolNode(
            type=node_dict.get("type"),
            in_stock=node_dict.get("in_stock", False),
            is_chemical=node_dict.get("is_chemical", False),
            smiles=node_dict.get("smiles", ""),
            children=parsed_children,
        )
    elif node_type == "reaction":
        metadata_dict = node_dict.get("metadata", {})
        parsed_metadata = ReactionMetadata(
            classification=metadata_dict.get("classification", "N/A"),
            policy_probability=metadata_dict.get("policy_probability", 0.0),
        )
        return ReactionNode(
            type=node_dict.get("type"),
            smiles=node_dict.get("smiles", ""),
            metadata=parsed_metadata,
            children=parsed_children,
        )
    return None


def parse_route_dict(route_dict: dict) -> Route:
    score_data = route_dict["scores"]
    parsed_score = Score(
        state_score=score_data["state score"],
        n_reactions=score_data["number of reactions"],
        n_precursors=score_data["number of pre-cursors"],
        n_precursors_in_stock=score_data["number of pre-cursors in stock"],
        avg_template_occurance=score_data["average template occurrence"],
    )

    parsed_root_node = parse_node(route_dict)

    if parsed_root_node is None:
        raise ValueError("Could not parse the root node from the 'dict' key.")

    return Route(score=parsed_score, root_node=parsed_root_node)
