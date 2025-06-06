from typing import List, Union

from pydantic import BaseModel, Field

Node = Union["MolNode", "ReactionNode"]


class ReactionMetadata(BaseModel):
    """
    Metadata for a reaction node in the retrosynthesis route.

    Attributes:
        classification (str): Classification of the reaction.
        policy_probability (float): Probability associated with the policy for this reaction.
    """

    classification: str
    policy_probability: float


class Score(BaseModel):
    """
    Score for a retrosynthesis route.

    Attributes:
        state_score (float): Overall score for the retrosynthesis state.
        n_reactions (int): Number of reactions in the route.
        n_precursors (int): Number of precursors in the route.
        n_precursors_in_stock (int): Number of precursors that are in stock.
        avg_template_occurance (float): Average occurrence of templates in the route.
    """

    state_score: float
    n_reactions: int
    n_precursors: int
    n_precursors_in_stock: int
    avg_template_occurance: float


class MolNode(BaseModel):
    """
    Represents a molecular node in the retrosynthesis route.

    Attributes:
        type (str): Type of the node, should be "mol".
        in_stock (bool): Indicates if the molecule is in stock.
        is_chemical (bool): Indicates if the node represents a chemical.
        smiles (str): SMILES representation of the molecule.
        children (List[Node]): List of child nodes, which can be either molecular or reaction nodes.
    """

    type: str
    in_stock: bool
    is_chemical: bool
    smiles: str
    children: List[Node] = Field(default=list)


class ReactionNode(BaseModel):
    """
    Represents a reaction node in the retrosynthesis route.

    Attributes:
        type (str): Type of the node, should be "reaction".
        smiles (str): SMILES representation of the reaction.
        metadata (ReactionMetadata): Metadata associated with the reaction.
        children (List[Node]): List of child nodes, which can be either molecular or reaction nodes.
    """

    type: str
    smiles: str
    metadata: ReactionMetadata
    children: List[Node] = Field(default=list)


class Route(BaseModel):
    """
    Represents a retrosynthesis route.

    Attributes:
        id (str): Unique identifier for the route.
        target_smiles (str): SMILES representation of the target molecule.
        scores (Score): Score associated with the route.
        root_node (Node): The root node of the retrosynthesis route, which can be a molecular or reaction node.
    """

    score: Score
    root_node: Node


class RouteCollection(BaseModel):
    """
    Represents a collection of retrosynthesis routes.

    Attributes:
        routes (List[Route]): List of retrosynthesis routes.
    """

    routes: List[Route] = Field(default_factory=list)

    def __getitem__(self, index: int) -> Route:
        return self.routes[index]

    def __len__(self) -> int:
        return len(self.routes)

    def __iter__(self):
        return iter(self.routes)


MolNode.model_rebuild()
ReactionNode.model_rebuild()
