from .models import FunctionDefinition, PromptDefinition
import json
from pydantic import ValidationError, BaseModel


def warn_extra_fields(item: dict, model: type[BaseModel]) -> None:
    """
    Warn about unexpected fields that will be ignored.

    Args:
        item: Input dictionary to inspect.
        model: Pydantic model defining the expected fields.
    """
    expected_fields = set(model.model_fields.keys())
    received_fields = set(item.keys())
    difference = received_fields - expected_fields
    if difference:
        fields_text = ", ".join(sorted(difference))
        print(f"**Warning: unexpected fields ignored: {fields_text}**\n")


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    """
    Load and validate function definitions from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        A list of validated function definitions.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValidationError: If the JSON structure does not match the models.
        ValueError: If the JSON root is not a list or its elements
        are not objects.
    """
    try:
        with open(path, "r") as file:
            data = json.load(file)
            if not isinstance(data, list):
                raise ValueError("The JSON root must be a list.")
            validated_functions = []
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError("The JSON elements must be dictionaries.")
                warn_extra_fields(item, FunctionDefinition)
                function = FunctionDefinition(**item)
                validated_functions.append(function)
            return validated_functions
    except FileNotFoundError as error:
        print(f"Error: couldn't find the required json file in {path}")
        print(error)
        raise
    except json.JSONDecodeError as error:
        print("Error: the json file is not valid")
        print(f"Line {error.lineno}, column {error.colno}: {error.msg}")
        raise
    except ValidationError as error:
        print("Error: the JSON structure is not valid.")
        print(error)
        raise
    except ValueError as error:
        print(f"Error: {error}")
        raise


def load_prompts(path: str) -> list[PromptDefinition]:
    """
    Load and validate prompts from a JSON file.

    Args:
        path: Path to the JSON file containing the prompts.

    Returns:
        A list of validated prompt definitions.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValidationError: If the JSON structure does not match the model.
        ValueError: If the JSON root is not a list or its elements
        are not objects.
    """
    try:
        with open(path, "r") as file:
            data = json.load(file)
            if not isinstance(data, list):
                raise ValueError("The JSON root must be a list.")
            validated_prompts = []
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError("The JSON elements must be dictionaries.")
                warn_extra_fields(item, PromptDefinition)
                prompt = PromptDefinition(**item)
                validated_prompts.append(prompt)
            return validated_prompts
    except FileNotFoundError as error:
        print(f"Error: couldn't find the required json file in {path}")
        print(error)
        raise
    except json.JSONDecodeError as error:
        print("Error: the json file is not valid")
        print(f"Line {error.lineno}, column {error.colno}: {error.msg}")
        raise
    except ValidationError as error:
        print("Error: the JSON structure is not valid.")
        print(error)
        raise
    except ValueError as error:
        print(f"Error: {error}")
        raise
