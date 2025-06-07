import io
from typing import Literal

from PIL import Image as PilImage, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw

from src.reagentai.models.retrosynthesis import (
    MolNode,
    ReactionMetadata,
    ReactionNode,
    Route,
    ScoreData,
)


### Helper classes and functions for processing retrosynthesis routes
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
        "average template occurrence": "avg_template_occurence",
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

    node_type = node_dict.get("type")
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
        node_data["in_stock"] = node_dict.get("in_stock", False)
        node_data["is_chemical"] = node_dict.get("is_chemical", True)
        parsed_node = MolNode(**node_data)
        state.processed_mol_nodes.append(parsed_node)
    elif node_type == "reaction":
        metadata = ReactionMetadata(
            classification=node_dict.get("classification", ""),
            policy_probability=node_dict.get("policy_probability", 0.0),
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


### Image processing functions for retrosynthesis routes
### Copied and refined from the AiZynthFinder project
def crop_image(img: PilImage, margin: int = 20) -> PilImage:
    """
    Crops the image to remove white borders and adds a margin around the cropped area.

    Args:
        img: The PIL Image to crop.
        margin: The margin to add around the cropped area.
    Returns:
        A new PIL Image with the cropped content and added margin.
    """
    non_white_pixels = PilImage.eval(img, lambda x: 255 - x).getbbox()
    if not non_white_pixels:
        return img
    cropped = img.crop(non_white_pixels)
    out = PilImage.new("RGB", (cropped.width + 2 * margin, cropped.height + 2 * margin), "white")
    out.paste(cropped, (margin, margin))
    return out


def draw_rounded_rectangle(
    img: PilImage.Image,
    color: str,
    arc_size: int = 20,
    width: int = 2,
) -> PilImage.Image:
    """
    Draws a rounded rectangle border on a copy of the given image.

    Args:
        img: The PIL Image to draw on.
        color: Color of the rectangle border.
        arc_size: Diameter of the corner arcs.
        border_width: Width of the border line.

    Returns:
        A new PIL Image with the rounded rectangle border.
    """

    copy = img.copy()
    draw = ImageDraw.Draw(copy)
    x0, y0, x1, y1 = 0, 0, copy.width - 1, copy.height - 1
    arc_size_half = arc_size // 2

    # Arcs for corners
    draw.arc((x0, y0, x0 + arc_size, y0 + arc_size), 180, 270, color, width=width)
    draw.arc((x1 - arc_size, y0, x1, y0 + arc_size), 270, 360, color, width=width)
    draw.arc((x1 - arc_size, y1 - arc_size, x1, y1), 0, 90, color, width=width)
    draw.arc((x0, y1 - arc_size, x0 + arc_size, y1), 90, 180, color, width=width)

    # Lines connecting arcs
    draw.line((x0 + arc_size_half, y0, x1 - arc_size_half, y0), fill=color, width=width)  # Top
    draw.line((x1, y0 + arc_size_half, x1, y1 - arc_size_half), fill=color, width=width)  # Right
    draw.line((x0 + arc_size_half, y1, x1 - arc_size_half, y1), fill=color, width=width)  # Bottom
    draw.line((x0, y0 + arc_size_half, x0, y1 - arc_size_half), fill=color, width=width)  # Left
    return copy


def molecules_to_images(
    mols: list[MolNode],
    in_stock_colors: dict[bool, str],
    size: int = 400,
) -> list[PilImage.Image]:
    """
    Converts a list of MolNode objects to a list of PIL images with rounded rectangles.

    Args:
        mols (list[MolNode]): List of MolNode objects to convert.
        in_stock_colors (dict[bool, str]): Dictionary mapping in-stock status to colors.
        size (int): Size of the images to generate.
    Returns:
        list[PilImage.Image]: List of PIL images with rounded rectangles.
    """

    rd_mols = [Chem.MolFromSmiles(mol.smiles) for mol in mols]
    grid_img = Draw.MolsToGridImage(rd_mols, molsPerRow=len(mols), subImgSize=(size, size))

    if isinstance(grid_img, bytes):
        grid_img = PilImage.open(io.BytesIO(grid_img))

    images = []
    for idx, mol in enumerate(mols):
        image_obj = grid_img.crop((size * idx, 0, size * (idx + 1), size))
        image_obj = crop_image(image_obj)
        frame_color = in_stock_colors[mol.in_stock]
        images.append(draw_rounded_rectangle(image_obj, frame_color))

    return images


class RouteImageFactory:
    """
    Factory class to create an image representation of a retrosynthetic route.
    This class builds a tree structure from the route's molecular and reaction nodes,
    calculates their effective sizes, positions them, and draws the final image.
    """

    def __init__(self, route: Route, margin: int = 100, mol_size: int = 400):
        in_stock_colors = {True: "green", False: "orange"}
        self.margin = margin
        self.mol_size = mol_size
        self.font = ImageFont.load_default()

        images = molecules_to_images(
            mols=route.mol_nodes,
            in_stock_colors=in_stock_colors,
            size=self.mol_size,
        )
        self._image_lookup = {
            mol.smiles: img for mol, img in zip(route.mol_nodes, images, strict=True)
        }

        self._reaction_lookup = {react.node_id: react for react in route.reaction_nodes}
        self._mol_lookup = {mol.node_id: mol for mol in route.mol_nodes}

        root_mol_node = self._mol_lookup.get(route.root_node_id)
        self._mol_tree = self._build_plot_tree(root_mol_node)

        self._add_effective_size(self._mol_tree)
        pos0 = (
            self._mol_tree["eff_width"] - self._mol_tree["image"].width + self.margin,
            int(self._mol_tree["eff_height"] * 0.5) - int(self._mol_tree["image"].height * 0.5),
        )
        self._add_pos(self._mol_tree, pos0)

        self.image = PilImage.new(
            "RGB",
            (self._mol_tree["eff_width"] + self.margin, self._mol_tree["eff_height"]),
            "white",
        )
        self._draw = ImageDraw.Draw(self.image)
        self._make_image(self._mol_tree)
        self.image = crop_image(self.image, margin=40)

    def _build_plot_tree(self, current_mol_node: MolNode) -> dict | None:
        """Recursively builds a tree structure for plotting, starting from a MolNode object."""

        tree_dict = {
            "smiles": current_mol_node.smiles,
            "image": self._image_lookup[current_mol_node.smiles],
            "id": current_mol_node.node_id,
        }

        children_trees = []

        if current_mol_node.children:
            reaction_node_id = current_mol_node.children[0]
            reaction_node_obj = self._reaction_lookup.get(reaction_node_id)

            for precursor_mol_node_id in reaction_node_obj.children:
                precursor_mol_node_obj = self._mol_lookup.get(precursor_mol_node_id)
                child_tree = self._build_plot_tree(precursor_mol_node_obj)
                children_trees.append(child_tree)

        tree_dict["children"] = children_trees
        return tree_dict

    def _add_effective_size(self, tree_dict: dict):
        """Recursively calculates the effective size of the tree dictionary."""

        children = tree_dict.get("children", [])
        for child in children:
            self._add_effective_size(child)

        if children:
            tree_dict["eff_height"] = sum(c["eff_height"] for c in children) + self.margin * (
                len(children) - 1
            )
            tree_dict["eff_width"] = (
                max(c["eff_width"] for c in children) + tree_dict["image"].width + self.margin
            )
        else:
            tree_dict["eff_height"] = tree_dict["image"].height
            tree_dict["eff_width"] = tree_dict["image"].width + self.margin

    def _add_pos(self, tree_dict: dict, pos: tuple[int, int]):
        """Recursively adds position information to the tree dictionary."""

        tree_dict["left"] = pos[0]
        tree_dict["top"] = pos[1]
        children = tree_dict.get("children")
        if not children:
            return
        mid_y = pos[1] + int(tree_dict["image"].height * 0.5)
        children_height = sum(c["eff_height"] for c in children) + self.margin * (
            len(children) - 1
        )
        childen_leftmost = pos[0] - self.margin - max(c["image"].width for c in children)
        child_y = mid_y - int(children_height * 0.5)
        for child in children:
            y_adjust = int((child["eff_height"] - child["image"].height) * 0.5)
            self._add_pos(child, (childen_leftmost, child_y + y_adjust))
            child_y += self.margin + child["eff_height"]

    def _make_image(self, tree_dict: dict):
        """Draws the molecule image and its connections on the canvas."""

        self.image.paste(tree_dict["image"], (tree_dict["left"], tree_dict["top"]))
        self._draw_number_on_frame(tree_dict)
        children = tree_dict.get("children")
        if not children:
            return
        children_right = max(c["left"] + c["image"].width for c in children)
        mid_x = children_right + int(0.5 * (tree_dict["left"] - children_right))
        mid_y = tree_dict["top"] + int(tree_dict["image"].height * 0.5)
        self._draw.line((tree_dict["left"], mid_y, mid_x, mid_y), fill="black", width=3)
        for child in children:
            self._make_image(child)
            child_mid_y = child["top"] + int(0.5 * child["image"].height)
            self._draw.line(
                (
                    mid_x,
                    mid_y,
                    mid_x,
                    child_mid_y,
                    child["left"] + child["image"].width,
                    child_mid_y,
                ),
                fill="black",
                width=3,
            )
        self._draw.ellipse(
            (mid_x - 8, mid_y - 8, mid_x + 8, mid_y + 8), fill="black", outline="black"
        )

    def _draw_number_on_frame(self, tree_dict: dict):
        """Draws the molecule ID as black text on the top-left corner of its frame."""

        image_x, image_y = tree_dict["left"], tree_dict["top"]
        mol_id = tree_dict["id"]
        text = str(mol_id)

        # Define a simple padding from the corner
        padding = 8
        text_pos = (image_x + padding, image_y + padding)

        # Draw the text directly onto the canvas with the new color
        self._draw.text(
            text_pos,
            text,
            font=self.font,
            fill="black",
        )
