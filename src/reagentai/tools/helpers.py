import io
import shutil
import tempfile
from typing import Generator, Literal

from PIL import Image as PilImage, ImageColor as PilColor, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw

from src.reagentai.models.retrosynthesis import (
    MolNode,
    Node,
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
        if node_type == "mol":
            self.mol_counter += 1
            return f"mol-{self.mol_counter}"
        elif node_type == "reaction":
            self.react_counter += 1
            return f"react-{self.react_counter}"
        else:
            raise ValueError(f"Unknown node type for ID generation: {node_type}")


def parse_route_score(score_dict: dict) -> ScoreData:
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
    node_type = node_dict.get("type")
    children_data = node_dict.get("children", [])

    current_node_id = state.get_next_id(node_type)

    child_ids = []
    for child_dict in children_data:
        child_id = parse_node(child_dict, state)
        child_ids.append(child_id)

    node_data = {
        "id": current_node_id,
        "smiles": node_dict["smiles"],
        "children": child_ids,
    }

    if node_type == "mol":
        node_data["in_stock"] = node_dict.get("in stock", False)
        node_data["is_chemical"] = node_dict.get("is chemical", True)
        parsed_node = MolNode(**node_data)
        state.processed_mol_nodes.append(parsed_node)
    elif node_type == "reaction":
        metadata = ReactionMetadata(
            classification=node_dict.get("classification", ""),
            policy_probability=node_dict.get("policy probability", 0.0),
        )
        node_data["metadata"] = metadata
        parsed_node = ReactionNode(**node_data)
        state.processed_reaction_nodes.append(parsed_node)

    return current_node_id


def parse_route_dict(route_dict: dict) -> Route:
    score_data = parse_route_score(route_dict["scores"])
    processing_state = NodeProcessingState()

    root_node_id = parse_node(route_dict, processing_state)

    return Route(
        score_data=score_data,
        root_node_id=root_node_id,
        mol_nodes=processing_state.processed_mol_nodes,
        reaction_nodes=processing_state.processed_reaction_nodes,
    )


def crop_image(img: PilImage, margin: int = 20) -> PilImage:
    non_white_pixels = PilImage.eval(img, lambda x: 255 - x).getbbox()
    if not non_white_pixels:
        return img
    cropped = img.crop(non_white_pixels)
    out = PilImage.new("RGB", (cropped.width + 2 * margin, cropped.height + 2 * margin), "white")
    out.paste(cropped, (margin, margin))
    return out


def draw_rounded_rectangle(img: PilImage.Image, color: str, arc_size: int = 20) -> PilImage:
    copy = img.copy()
    draw = ImageDraw.Draw(copy)
    x0, y0, x1, y1 = 0, 0, copy.width - 1, copy.height - 1
    arc_size_half = arc_size // 2
    draw.arc((x0, y0, x0 + arc_size, y0 + arc_size), 180, 270, color, width=2)
    draw.arc((x1 - arc_size, y0, x1, y0 + arc_size), 270, 360, color, width=2)
    draw.arc((x1 - arc_size, y1 - arc_size, x1, y1), 0, 90, color, width=2)
    draw.arc((x0, y1 - arc_size, x0 + arc_size, y1), 90, 180, color, width=2)
    draw.line((x0 + arc_size_half, y0, x1 - arc_size_half, y0), fill=color, width=2)
    draw.line((x1, y0 + arc_size_half, x1, y1 - arc_size_half), fill=color, width=2)
    draw.line((x0 + arc_size_half, y1, x1 - arc_size_half, y1), fill=color, width=2)
    draw.line((x0, y0 + arc_size_half, x0, y1 - arc_size_half), fill=color, width=2)
    return copy


def molecules_to_images(
    mols: dict[int, str], frame_colors: list[str], size: int = 400
) -> list[PilImage.Image]:
    rd_mols = [Chem.MolFromSmiles(smiles) for smiles in mols.values()]
    grid_img = Draw.MolsToGridImage(rd_mols, molsPerRow=len(mols), subImgSize=(size, size))

    if isinstance(grid_img, bytes):
        grid_img = PilImage.open(io.BytesIO(grid_img))
    images = []

    for idx, frame_color in enumerate(frame_colors):
        image_obj = grid_img.crop((size * idx, 0, size * (idx + 1), size))
        image_obj = crop_image(image_obj)
        images.append(draw_rounded_rectangle(image_obj, frame_color))

    return images


class RouteImageFactory:
    def __init__(self, route: Route, margin: int = 100, mol_size: int = 400):
        in_stock_colors = {True: "green", False: "orange"}
        self.margin = margin
        self.mol_size = mol_size
        font_size = max(16, self.mol_size // 15)
        self.font = ImageFont.load_default()

        self._smiles_lookup = dict(self._extract_molecules_from_node(route.root_node))

        images = molecules_to_images(
            self._smiles_lookup.keys,
            [in_stock_colors[val] for val in self._smiles_lookup.values()],
            size=self.mol_size,
        )
        self._image_lookup = dict(zip(self._smiles_lookup.keys(), images, strict=True))

        self._mol_tree = self._build_plot_tree(route.root_node)
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
        # self.image = crop_image(self.image, margin=0)

    def _extract_molecules_from_node(self, node: Node) -> Generator[tuple[str, bool], None, None]:
        if isinstance(node, MolNode):
            yield (node.smiles, node.in_stock)

        for child in node.children:
            self._extract_molecules_from_node(child)

    def _build_plot_tree(self, node: Node) -> dict | None:
        if not isinstance(node, MolNode):
            return None

        tree_dict = {
            "smiles": node.smiles,
            "image": self._image_lookup[node.smiles],
            "id": node.id_in_route,
        }

        children_trees = []
        if node.children:
            reaction_node = node.children[0]

            if isinstance(reaction_node, ReactionNode):
                for reactant_node in reaction_node.children:
                    child_tree = self._build_plot_tree(reactant_node)
                    if child_tree:
                        children_trees.append(child_tree)

        tree_dict["children"] = children_trees
        return tree_dict

    def _add_effective_size(self, tree_dict: dict):
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

    def _draw_number_on_frame(self, tree_dict: dict):  # REVISED METHOD
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
            fill="black",  # Changed from "white"
        )
