from .models import FunctionDefinition, PromptDefinition
import json
from pydantic import ValidationError


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
    """
    try:
        with open(path, "r") as file:
            data = json.load(file)
            validated_functions = []
            for item in data:
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
        print("Error: the information structure in the json file is not valid")
        print(error)
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
    """
    try:
        with open(path, "r") as file:
            data = json.load(file)
            validated_prompts = []
            for item in data:
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
        print("Error: the information structure in the json file is not valid")
        print(error)
        raise
