from pydantic import BaseModel


class ImageOutput(BaseModel):
    """
    Represents the output of a tool that generates an image.

    Attributes:
        file_path (str): The file path to the image output.
        title (str): A title for the image output.
    """

    file_path: str
    title: str
