from dotenv import load_dotenv
from pydantic_ai import Agent
from constants import INSTRUCTIONS_PATH


def get_instructions(instruction_file_path: str) -> str:
    """
    Read instructions from a file.

    Returns:
        str: The instructions read from the file.
    """

    with open(instruction_file_path, "r") as instructions_file:
        instructions = instructions_file.read()

    return instructions


def main():
    load_dotenv()
    reagent = Agent(
        "google-gla:gemini-2.0-flash",
        instructions=get_instructions(INSTRUCTIONS_PATH),
    )
    result = reagent.run_sync(
        "Introduce yourself, and tell me how to synthesize a sodium chloride"
    )
    print(result.output)

if __name__ == "__main__":
    main()
