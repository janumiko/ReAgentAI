"""
Example retrosynthesis analysis using AiZynthFinder library.
This module demonstrates how to perform retrosynthetic analysis on target molecules.
"""

from typing import Any
from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.analysis import RouteCollection


class AiZynthFinderWrapper:
    def __init__(self, config_path: str) -> None:
        """
        Initialize the AiZynthFinder with the given configuration file.

        Args:
            config_path (str): Path to the configuration file for AiZynthFinder.
        """
        self.finder = AiZynthFinder(configfile=config_path)

        self.finder.stock.select("zinc")
        self.finder.expansion_policy.select("uspto")
        self.finder.filter_policy.select("uspto")

    def perform_tree_search(
        self, target_smile: str
    ) -> tuple[RouteCollection, dict[str, Any]]:
        """
        Perform a retrosynthetic tree search for the given target SMILES.

        Args:
            target_smile (str): SMILES representation of the target molecule.

        Returns:
            tuple[RouteCollection, dict[str, Any]]: A tuple containing the routes and statistics.
        """

        self.finder.target_smiles = target_smile
        self.finder.tree_search()
        self.finder.build_routes()

        routes = self.finder.routes
        statistics = self.finder.extract_statistics()

        return (routes, statistics)


def prettify_route(route_data: dict, indent: int = 0) -> str:
    """
    Prettify a single retrosynthesis route for better readability.

    Args:
        route_data (dict): Route data from AiZynthFinder
        indent (int): Current indentation level

    Returns:
        str: Formatted string representation of the route
    """
    output = []
    prefix = "  " * indent

    if route_data.get("type") == "mol":
        # Format molecule
        smiles = route_data.get("smiles", "Unknown")
        in_stock = route_data.get("in_stock", False)
        stock_status = "✓ In Stock" if in_stock else "✗ Not in Stock"

        output.append(f"{prefix}🧪 Molecule: {smiles}")
        output.append(f"{prefix}   Status: {stock_status}")

        # Show scores if available
        if "scores" in route_data:
            for score_name, score_value in route_data["scores"].items():
                output.append(f"{prefix}   {score_name.title()}: {score_value:.4f}")

    elif route_data.get("type") == "reaction":
        # Format reaction
        smiles = route_data.get("smiles", "Unknown")
        output.append(f"{prefix}⚗️  Reaction: {smiles}")

        # Show metadata if available
        if "metadata" in route_data:
            metadata = route_data["metadata"]
            if "classification" in metadata:
                output.append(
                    f"{prefix}   Classification: {metadata['classification']}"
                )
            if "policy_probability" in metadata:
                prob = metadata["policy_probability"]
                output.append(f"{prefix}   Probability: {prob:.4f}")
            if "library_occurence" in metadata:
                output.append(
                    f"{prefix}   Library Occurrences: {metadata['library_occurence']}"
                )

    # Process children recursively
    if "children" in route_data and route_data["children"]:
        output.append(f"{prefix}└── Precursors:")
        for i, child in enumerate(route_data["children"]):
            if i > 0:
                output.append("")  # Add spacing between children
            output.append(prettify_route(child, indent + 1))

    return "\n".join(output)


def print_retrosynthesis_results(
    routes: RouteCollection, stats: dict[str, Any], target: str
) -> None:
    """
    Print prettified retrosynthesis results with explanatory context.

    Args:
        routes (RouteCollection): Routes from AiZynthFinder
        stats (dict): Statistics from the search
        target (str): Target molecule SMILES
    """
    print("=" * 80)
    print("🎯 RETROSYNTHESIS ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Target Molecule: {target}")
    print("📝 Note: 'In Stock' refers to availability in ZINC database for research,")
    print("    not commercial pharmaceutical availability.")
    print()

    # Print key statistics
    print("📊 Search Statistics:")
    print(f"  Search Time: {stats.get('search_time', 0):.2f} seconds")
    print(f"  Routes Found: {stats.get('number_of_routes', 0)}")
    print(f"  Solved Routes: {stats.get('number_of_solved_routes', 0)}")
    print(f"  Top Score: {stats.get('top_score', 0):.4f}")
    print(f"  Is Solved: {'✓' if stats.get('is_solved', False) else '✗'}")

    # Show precursor availability summary
    precursors_in_stock = stats.get("precursors_in_stock", "")
    precursors_not_in_stock = stats.get("precursors_not_in_stock", "")

    if precursors_in_stock:
        print(f"  Precursors Available: {precursors_in_stock}")
    if precursors_not_in_stock:
        print(f"  Precursors Needed: {precursors_not_in_stock}")
    print()

    # Print the best route(s)
    routes_data = routes.dict_with_scores()
    if not routes_data:
        print("❌ No viable routes found.")
        return

    print(
        f"🛤️  Best Route (Score: {routes_data[0].get('scores', {}).get('state score', 0):.4f}):"
    )
    print("-" * 60)
    print(prettify_route(routes_data[0]))

    # Show synthesis complexity
    steps = stats.get("number_of_steps", 0)
    precursors = stats.get("number_of_precursors", 0)
    available_precursors = stats.get("number_of_precursors_in_stock", 0)

    print()
    print("🧬 Synthesis Summary:")
    print(f"  Number of Steps: {steps}")
    print(f"  Total Precursors: {precursors}")
    print(f"  Available Precursors: {available_precursors}/{precursors}")

    if available_precursors == precursors:
        print("  ✅ All starting materials are available!")
    else:
        print("  ⚠️  Some starting materials may need synthesis or sourcing.")


def main():
    config_path = "data/config.yml"

    # Initialize the AiZynthFinder with default settings
    finder = AiZynthFinderWrapper(config_path)

    # Define the target molecule for retrosynthesis
    target = "CC(=O)Nc1ccc(O)cc1"  # Example: Acetaminophen (Paracetamol)

    print("🔍 Starting retrosynthesis analysis...")
    print(f"Target: Paracetamol/Acetaminophen ({target})")
    print()

    routes, stats = finder.perform_tree_search(target)
    print_retrosynthesis_results(routes, stats, target)
