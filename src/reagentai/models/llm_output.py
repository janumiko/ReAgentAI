from pydantic import BaseModel, Field


class TextOutput(BaseModel):
    """
    Represents the output of a language model as text.

    Attributes:
        text (str): The text output from the language model.
    """

    text: str

    def to_message(self) -> dict:
        """
        Converts the text output to a message format suitable for display.

        Returns:
            dict: A dictionary containing the role and content of the message.
        """
        return {"role": "assistant", "content": self.text}


class ImageOutput(BaseModel):
    """
    Represents the output of a language model as an image.

    Attributes:
        file_path (str): The file path to the image output.
    """

    file_path: str

    def to_message(self) -> dict:
        """
        Converts the image output to a message format suitable for display.

        Returns:
            dict: A dictionary containing the role and content of the message.
        """
        return {"role": "assistant", "content": {"path": self.file_path}}


class MultipleOutputs(BaseModel):
    """
    Represents a collection of outputs from a language model.

    Attributes:
        outputs (list[TextOutput | ImageOutput]): A list of outputs that can be either text or image.
    """

    outputs: list[TextOutput | ImageOutput] = Field(default_factory=list)

    def to_message(self) -> list[dict]:
        """
        Converts the multiple outputs to a message format suitable for display.

        Returns:
            list[dict]: A list of dictionaries containing the role and content of each message.
        """
        return [output.to_message() for output in self.outputs]
