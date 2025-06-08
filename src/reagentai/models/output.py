from pydantic import BaseModel


class ImageOutput(BaseModel):
    """
    Represents the output of a tool that generates an image.

    Attributes:
        file_path (str): The file path to the image output.
        description (str): A description or title for the image output.
    """

    file_path: str
    description: str
