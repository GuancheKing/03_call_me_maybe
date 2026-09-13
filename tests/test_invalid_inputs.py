import json
import os
import tempfile

from pydantic import ValidationError

from src.parser import (
    load_function_definitions,
    load_prompts,
)


def write_temp_file(content: str) -> str:
    """
    Create a temporary JSON file with the provided content.

    Args:
        content: Text content to write to the temporary file.

    Returns:
        Path to the created temporary file.
    """
    temporary_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
    )
    temporary_file.write(content)
    temporary_file.close()

    return temporary_file.name


def test_broken_json() -> None:
    """
    Check that malformed JSON raises JSONDecodeError.
    """
    path = write_temp_file(
        '[{"name": "fn_test",]'
    )

    try:
        try:
            load_function_definitions(path)
            raise AssertionError(
                "Expected JSONDecodeError for malformed JSON."
            )
        except json.JSONDecodeError:
            print("Broken JSON: passed")
    finally:
        os.remove(path)


def test_root_not_list() -> None:
    """
    Check that a JSON root that is not a list raises ValueError.
    """
    path = write_temp_file(
        '{"name": "fn_test"}'
    )

    try:
        try:
            load_function_definitions(path)
            raise AssertionError(
                "Expected ValueError when JSON root is not a list."
            )
        except ValueError:
            print("Root not list: passed")
    finally:
        os.remove(path)


def test_element_not_dictionary() -> None:
    """
    Check that non-object list elements raise ValueError.
    """
    path = write_temp_file(
        '["not a dictionary"]'
    )

    try:
        try:
            load_function_definitions(path)
            raise AssertionError(
                "Expected ValueError for non-dictionary element."
            )
        except ValueError:
            print("Element not dictionary: passed")
    finally:
        os.remove(path)


def test_function_without_name() -> None:
    """
    Check that a function without a name raises ValidationError.
    """
    path = write_temp_file(
        """
        [
            {
                "description": "Test function.",
                "parameters": {},
                "returns": {
                    "type": "string"
                }
            }
        ]
        """
    )

    try:
        try:
            load_function_definitions(path)
            raise AssertionError(
                "Expected ValidationError for missing function name."
            )
        except ValidationError:
            print("Function without name: passed")
    finally:
        os.remove(path)


def test_invalid_parameter_structure() -> None:
    """
    Check that an invalid parameter definition raises ValidationError.
    """
    path = write_temp_file(
        """
        [
            {
                "name": "fn_test",
                "description": "Test function.",
                "parameters": {
                    "value": "string"
                },
                "returns": {
                    "type": "string"
                }
            }
        ]
        """
    )

    try:
        try:
            load_function_definitions(path)
            raise AssertionError(
                "Expected ValidationError for invalid parameter."
            )
        except ValidationError:
            print("Invalid parameter structure: passed")
    finally:
        os.remove(path)


def test_prompt_without_prompt_field() -> None:
    """
    Check that a prompt without the prompt field raises ValidationError.
    """
    path = write_temp_file(
        """
        [
            {
                "message": "Hello"
            }
        ]
        """
    )

    try:
        try:
            load_prompts(path)
            raise AssertionError(
                "Expected ValidationError for missing prompt field."
            )
        except ValidationError:
            print("Prompt without prompt field: passed")
    finally:
        os.remove(path)


def test_missing_file() -> None:
    """
    Check that a missing file raises FileNotFoundError.
    """
    path = "/tmp/call_me_maybe_file_that_does_not_exist.json"

    try:
        load_prompts(path)
        raise AssertionError(
            "Expected FileNotFoundError for missing file."
        )
    except FileNotFoundError:
        print("Missing file: passed")


test_broken_json()
test_root_not_list()
test_element_not_dictionary()
test_function_without_name()
test_invalid_parameter_structure()
test_prompt_without_prompt_field()
test_missing_file()

print()
print("All invalid input tests passed.")
