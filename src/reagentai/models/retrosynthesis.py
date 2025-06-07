from typing import Union

from pydantic import BaseModel, Field


Node = Union['MolNode', 'ReactionNode']


class ReactionMetadata(BaseModel):
    """
    Metadata for a reaction node in the retrosynthesis route.

    Attributes:
        classification (str): Classification of the reaction.
        policy_probability (float): Probability associated with the policy for this reaction.
    """

    classification: str
    policy_probability: float


class ScoreData(BaseModel):
    """
    Represents the scoring data for a retrosynthesis route.

    Attributes:
        state_score (float): Overall score for the retrosynthesis state.
        n_reactions (int): Number of reactions in the route.
        n_precursors (int): Number of precursors in the route.
        n_precursors_in_stock (int): Number of precursors that are in stock.
        avg_template_occurence (float): Average occurrence of templates in the route.
    """

    state_score: float
    n_reactions: int
    n_precursors: int
    n_precursors_in_stock: int
    avg_template_occurence: float


class NodeBase(BaseModel):
    """
    Represents a base node in a retrosynthesis tree.

    Attributes:
        id (str): Unique identifier for the node.
        smiles (str): SMILES string representing the chemical structure.
        children (list[str]): List of IDs of child nodes (default is an empty list).
    """

    id: str
    smiles: str
    children: list[str] = Field(default_factory=list)


class MolNode(NodeBase):
    """
    Represents a molecular node in a retrosynthesis graph.

    Attributes:
        in_stock (bool): Indicates whether the molecule is available in stock.
        is_chemical (bool): Specifies if the node represents a chemical entity.
    """

    in_stock: bool
    is_chemical: bool


class ReactionNode(NodeBase):
    """
    Represents a node in a retrosynthesis reaction tree.

    Attributes:
        metadata (ReactionMetadata): Metadata associated with the reaction.
    """

    metadata: ReactionMetadata

class Route(BaseModel):
    """
    Represents a retrosynthetic route consisting of molecular and reaction nodes.

    Attributes:
        score_data (ScoreData): The scoring information associated with the route.
        root_node_id (str): The identifier of the root node in the route.
        mols_nodes (list[MolNode]): List of molecular nodes in the route.
        reactions_nodes (list[ReactionNode]): List of reaction nodes in the route.
    """
    score_data: ScoreData
    root_node_id: str
    mol_nodes: list[MolNode] = Field(default_factory=list)
    reaction_nodes: list[ReactionNode] = Field(default_factory=list)


class RouteCollection(BaseModel):
    """
    Represents a collection of retrosynthesis routes.

    Attributes:
        routes (List[Route]): List of retrosynthesis routes.
        n_routes (int): Number of routes in the collection.
    """

    routes: list[Route] = Field(default_factory=list)
    n_routes: int = Field(default=0)

    def __len__(self) -> int:
        """
        Returns the number of routes in the collection.
        """
        return self.n_routes

    def __getitem__(self, index: int) -> Route:
        """
        Returns the route at the specified index.

        Args:
            index (int): Index of the route to retrieve.

        Returns:
            Route: The route at the specified index.
        """
        return self.routes[index]

    def __iter__(self):
        """
        Returns an iterator over the routes in the collection.
        """
        return iter(self.routes)


MolNode.model_rebuild()
ReactionNode.model_rebuild()
