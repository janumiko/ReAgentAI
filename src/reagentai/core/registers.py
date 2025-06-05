from pydantic_ai.tools import Tool, DocstringFormat


_registered_tools = []


def register_tool(takes_ctx: bool = False, docstring_format: DocstringFormat = "auto"):
    """
    Decorator to register a function as a tool.

    Args:
        name (str): The name of the tool.
    """

    def decorator(func):
        _registered_tools.append(
            Tool(
                function=func,
                takes_ctx=takes_ctx,
                name=func.__name__,
                description=func.__doc__,
                docstring_format=docstring_format,
            )
        )
        return func

    return decorator


def get_registered_tools():
    """
    Returns the dictionary of registered tools.
    """
    return _registered_tools
