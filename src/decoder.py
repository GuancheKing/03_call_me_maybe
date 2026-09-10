from enum import Enum
from pydantic import BaseModel
from .models import FunctionDefinition, ParameterDefinition


class CandidateStatus(Enum):
    """Represent the validation status of a generated candidate."""
    VALID_COMPLETE = "valid_complete"
    VALID_PREFIX = "valid_prefix"
    INVALID = "invalid"


class DecoderState(Enum):
    """Represent the possible states of the constrained decoder."""
    START = "start"
    OBJECT_OPEN = "object_open"
    KEY_OPEN = "key_open"
    KEY_TEXT = "key_text"
    KEY_CLOSE = "key_close"
    COLON = "colon"
    VALUE_OPEN = "value_open"
    FUNCTION_NAME = "function_name"
    VALUE_CLOSE = "value_close"
    OBJECT_CLOSE = "object_close"
    COMMA = "comma"
    PARAMETERS_OPEN = "parameters_open"
    PARAMETER_NAME = "parameter_name"
    PARAMETER_KEY_OPEN = "parameter_key_open"
    PARAMETER_KEY_CLOSE = "parameter_key_close"
    PARAMETER_COLON = "parameter_colon"
    PARAMETER_VALUE_OPEN = "parameter_value_open"
    PARAMETER_STRING_VALUE = "parameter_string_value"
    PARAMETER_STRING_ESCAPE = "parameter_string_escape"
    PARAMETER_STRING_UNICODE = "parameter_string_unicode"
    PARAMETER_NUMBER_VALUE = "parameter_number_value"
    PARAMETER_BOOLEAN_VALUE = "parameter_boolean_value"
    PARAMETER_VALUE_CLOSE = "parameter_value_close"
    PARAMETER_COMMA = "parameter_comma"
    PARAMETER_COMPLETE = "parameter_complete"
    COMPLETE = "complete"
    INVALID = "invalid"


class DecoderContext(BaseModel):
    """Store the current state and buffers used during decoding."""
    state: DecoderState
    expected_key: str = "name"
    key_buffer: str = ""
    function_name_buffer: str = ""
    parameter_name_buffer: str = ""
    parameter_names: list[str] = []
    current_parameter_type: str = ""
    parameter_value_buffer: str = ""
    used_parameter_names: list[str] = []
    unicode_escape_buffer: str = ""


def is_valid_prefix(candidate: str, target: str) -> bool:
    """
    Check whether a candidate is a valid prefix of a target string.

    Args:
        candidate: Generated text to validate.
        target: Expected target string.

    Returns:
        True if the target starts with the candidate, otherwise False.
    """
    return target.startswith(candidate)


def is_a_valid_number(value: str) -> bool:
    """
    Check whether a string represents a valid simple JSON number.

    Args:
        value: String representation of the number to validate.

    Returns:
        True if the value is a valid integer or decimal number,
        otherwise False.
    """
    # Exponential notation is not supported yet.
    if not value:
        return False
    has_decimal_point = False
    start_index = 1 if value[0] == "-" else 0
    if start_index < len(value):
        if value[start_index] == "0":
            if start_index == len(value) - 1:
                return True
            if value[start_index + 1] != ".":
                return False
    for index, char in enumerate(value):
        if index == 0 and char == "-":
            if len(value) == 1:
                return False
            continue
        if index > 0 and char == "-":
            return False
        if (
            (index == start_index and char == ".") or
            (index == len(value) - 1 and char == ".")
        ):
            return False
        if char == "." and not has_decimal_point:
            has_decimal_point = True
        elif has_decimal_point and char == ".":
            return False
        elif char not in "0123456789":
            return False
        continue
    return True


def check_candidates(
        candidate: str, allowed_names: list[str]
        ) -> CandidateStatus:
    """
    Validate a candidate against a list of allowed names.

    Args:
        candidate: Generated value or partial value to validate.
        allowed_names: List of valid names or values.

    Returns:
        The validation status of the candidate.
    """
    has_valid_prefix = False
    for function_name in allowed_names:
        if candidate == function_name:
            return CandidateStatus.VALID_COMPLETE
        if function_name.startswith(candidate):
            has_valid_prefix = True
    if has_valid_prefix:
        return CandidateStatus.VALID_PREFIX
    return CandidateStatus.INVALID


def get_function_parameters(
    function_name: str,
    function_definitions: list[FunctionDefinition]
) -> dict[str, ParameterDefinition]:
    """
    Get the parameter definitions for a function.

    Args:
        function_name: Name of the function to search for.
        function_definitions: Available function definitions.

    Returns:
        The parameter definitions of the selected function.

    Raises:
        ValueError: If the function is not found.
    """
    for each in function_definitions:
        if each.name == function_name:
            return each.parameters
    raise ValueError(
        f"Function '{function_name}' was not found in"
        " the available definitions."
        )


def next_state(
    current_state: DecoderState,
    char: str,
    context: DecoderContext,
    allowed_names: list[str]
) -> DecoderState:
    """
    Determine the next decoder state for a generated character.

    Args:
        current_state: Current decoder state.
        char: Character to validate.
        context: Current decoding context.
        allowed_names: Function names available for generation.

    Returns:
        The next decoder state, or INVALID if the character is not allowed.
    """
    if current_state == DecoderState.START and char == "{":
        return DecoderState.OBJECT_OPEN
    elif current_state == DecoderState.OBJECT_OPEN and char == '"':
        return DecoderState.KEY_OPEN
    elif (
        current_state == DecoderState.KEY_OPEN and
        char == context.expected_key[0]
    ):
        return DecoderState.KEY_TEXT
    elif (
        current_state == DecoderState.KEY_TEXT and
        context.expected_key.startswith(
            context.key_buffer + char
        )
    ):
        return DecoderState.KEY_TEXT
    elif (
        current_state == DecoderState.KEY_TEXT and
        context.key_buffer == context.expected_key and char == '"'
    ):
        return DecoderState.KEY_CLOSE
    elif current_state == DecoderState.KEY_CLOSE and char == ":":
        return DecoderState.COLON
    elif current_state == DecoderState.COLON:
        if context.expected_key == "name" and char == '"':
            return DecoderState.VALUE_OPEN
        if context.expected_key == "parameters" and char == '{':
            return DecoderState.PARAMETERS_OPEN
    elif current_state == DecoderState.VALUE_OPEN:
        candidate = context.function_name_buffer + char
        status = check_candidates(candidate, allowed_names)

        if status != CandidateStatus.INVALID:
            return DecoderState.FUNCTION_NAME

    elif current_state == DecoderState.FUNCTION_NAME:
        if (
            char == '"'
            and check_candidates(
                context.function_name_buffer,
                allowed_names
            ) == CandidateStatus.VALID_COMPLETE
        ):
            return DecoderState.VALUE_CLOSE

        candidate = context.function_name_buffer + char
        status = check_candidates(candidate, allowed_names)

        if status != CandidateStatus.INVALID:
            return DecoderState.FUNCTION_NAME
    elif current_state == DecoderState.VALUE_CLOSE and char == ",":
        return DecoderState.COMMA
    elif current_state == DecoderState.COMMA and char == '"':
        return DecoderState.KEY_OPEN
    elif current_state == DecoderState.PARAMETERS_OPEN and char == '"':
        return DecoderState.PARAMETER_KEY_OPEN
    elif current_state == DecoderState.PARAMETERS_OPEN and char == '}':
        if not context.parameter_names:
            return DecoderState.PARAMETER_COMPLETE
    elif current_state == DecoderState.PARAMETER_KEY_OPEN:
        candidate = context.parameter_name_buffer + char
        available_parameter_names = [
            name
            for name in context.parameter_names
            if name not in context.used_parameter_names
        ]
        status = check_candidates(candidate, available_parameter_names)
        if status == CandidateStatus.INVALID:
            return DecoderState.INVALID
        return DecoderState.PARAMETER_NAME
    elif current_state == DecoderState.PARAMETER_NAME:
        available_parameter_names = [
                    name
                    for name in context.parameter_names
                    if name not in context.used_parameter_names
                ]
        if (
            char == '"'
            and check_candidates(
                context.parameter_name_buffer,
                available_parameter_names
            ) == CandidateStatus.VALID_COMPLETE
        ):
            return DecoderState.PARAMETER_KEY_CLOSE

        candidate = context.parameter_name_buffer + char
        status = check_candidates(candidate, available_parameter_names)

        if status != CandidateStatus.INVALID:
            return DecoderState.PARAMETER_NAME
    elif current_state == DecoderState.PARAMETER_KEY_CLOSE and char == ':':
        return DecoderState.PARAMETER_COLON
    elif current_state == DecoderState.PARAMETER_COLON and char == '"':
        if context.current_parameter_type == "string":
            return DecoderState.PARAMETER_VALUE_OPEN
    # Scientific notation is not supported; valid JSON numbers using
    # exponent notation (e.g. 1e3) will be rejected by the decoder.
    elif (
        current_state == DecoderState.PARAMETER_COLON and
        char in '-0123456789'
    ):
        if context.current_parameter_type == "number":
            return DecoderState.PARAMETER_NUMBER_VALUE
    elif (
        current_state == DecoderState.PARAMETER_COLON and
        context.current_parameter_type == "boolean" and
        char in ("t", "f")
    ):
        return DecoderState.PARAMETER_BOOLEAN_VALUE
    elif current_state == DecoderState.PARAMETER_VALUE_OPEN:
        if char == "\\":
            return DecoderState.PARAMETER_STRING_ESCAPE
        if char != '"':
            return DecoderState.PARAMETER_STRING_VALUE
        return DecoderState.PARAMETER_VALUE_CLOSE
    elif current_state == DecoderState.PARAMETER_STRING_VALUE:
        if char == "\\":
            return DecoderState.PARAMETER_STRING_ESCAPE
        if char == '"':
            return DecoderState.PARAMETER_VALUE_CLOSE
        return DecoderState.PARAMETER_STRING_VALUE
    elif current_state == DecoderState.PARAMETER_STRING_ESCAPE:
        if char == "u":
            return DecoderState.PARAMETER_STRING_UNICODE
        if char in '"\\/bfnrt':
            return DecoderState.PARAMETER_STRING_VALUE
        return DecoderState.INVALID
    elif current_state == DecoderState.PARAMETER_STRING_UNICODE:
        if char not in "0123456789abcdefABCDEF":
            return DecoderState.INVALID
        if len(context.unicode_escape_buffer) == 3:
            return DecoderState.PARAMETER_STRING_VALUE
        return DecoderState.PARAMETER_STRING_UNICODE
    elif current_state == DecoderState.PARAMETER_NUMBER_VALUE:
        if (
            char in '0123456789' and
            context.parameter_value_buffer not in ("0", "-0")
        ):
            return DecoderState.PARAMETER_NUMBER_VALUE
        elif context.parameter_value_buffer == "-" and char == ".":
            return DecoderState.INVALID
        elif char == "." and "." not in context.parameter_value_buffer:
            return DecoderState.PARAMETER_NUMBER_VALUE
        elif char == ",":
            if is_a_valid_number(context.parameter_value_buffer):
                return DecoderState.PARAMETER_COMMA
            return DecoderState.INVALID
        elif char == "}":
            if is_a_valid_number(context.parameter_value_buffer):
                completed_parameter_names = (
                    context.used_parameter_names
                    + [context.parameter_name_buffer]
                )
                for name in context.parameter_names:
                    if name not in completed_parameter_names:
                        return DecoderState.INVALID
                return DecoderState.PARAMETER_COMPLETE
            return DecoderState.INVALID
    elif current_state == DecoderState.PARAMETER_BOOLEAN_VALUE:
        if (
            char == "," and
            check_candidates(
                context.parameter_value_buffer,
                ["true", "false"]
            ) == CandidateStatus.VALID_COMPLETE
        ):
            return DecoderState.PARAMETER_COMMA
        if (
            char == "}" and
                check_candidates(
                    context.parameter_value_buffer,
                    ["true", "false"]
                ) == CandidateStatus.VALID_COMPLETE
        ):
            completed_parameter_names = (
                context.used_parameter_names
                + [context.parameter_name_buffer]
            )
            for name in context.parameter_names:
                if name not in completed_parameter_names:
                    return DecoderState.INVALID
            return DecoderState.PARAMETER_COMPLETE
        candidate = context.parameter_value_buffer + char
        status = check_candidates(candidate, ["true", "false"])
        if status != CandidateStatus.INVALID:
            return DecoderState.PARAMETER_BOOLEAN_VALUE
    elif current_state == DecoderState.PARAMETER_VALUE_CLOSE:
        if char == ',':
            return DecoderState.PARAMETER_COMMA
        elif char == '}':
            completed_parameter_names = (
                                context.used_parameter_names
                                + [context.parameter_name_buffer]
                            )
            for name in context.parameter_names:
                if name not in completed_parameter_names:
                    return DecoderState.INVALID
            return DecoderState.PARAMETER_COMPLETE
    elif current_state == DecoderState.PARAMETER_COMMA and char == '"':
        return DecoderState.PARAMETER_KEY_OPEN
    elif current_state == DecoderState.PARAMETER_COMPLETE and char == '}':
        return DecoderState.COMPLETE
    return DecoderState.INVALID


def update_context(
        context: DecoderContext,
        char: str,
        allowed_names: list[str],
        function_definitions: list[FunctionDefinition]
) -> DecoderContext:
    """
    Update the decoder context after processing a character.

    Args:
        context: Current decoding context.
        char: Character being processed.
        allowed_names: Function names available for generation.
        function_definitions: Available function definitions.

    Returns:
        The updated decoder context.
    """
    previous_state = context.state
    new_state = next_state(context.state, char, context, allowed_names)
    context.state = new_state
    if new_state == DecoderState.KEY_TEXT:
        context.key_buffer += char
    if new_state == DecoderState.FUNCTION_NAME:
        context.function_name_buffer += char
    if new_state == DecoderState.COMMA:
        if context.expected_key == "name":
            context.expected_key = "parameters"
            context.key_buffer = ""
    if new_state == DecoderState.PARAMETERS_OPEN:
        parameters = get_function_parameters(
            context.function_name_buffer,
            function_definitions
        )
        context.parameter_names = list(parameters.keys())
    if new_state == DecoderState.PARAMETER_NAME:
        context.parameter_name_buffer += char
    if new_state == DecoderState.PARAMETER_COLON:
        parameters = get_function_parameters(
            context.function_name_buffer,
            function_definitions
        )
        context.current_parameter_type = (
            parameters[context.parameter_name_buffer].type
        )
    if new_state == DecoderState.PARAMETER_NUMBER_VALUE:
        context.parameter_value_buffer += char

    if new_state == DecoderState.PARAMETER_BOOLEAN_VALUE:
        context.parameter_value_buffer += char

    if new_state == DecoderState.PARAMETER_STRING_ESCAPE:
        context.parameter_value_buffer += char

    if (
        previous_state == DecoderState.PARAMETER_STRING_ESCAPE
        and new_state == DecoderState.PARAMETER_STRING_UNICODE
    ):
        # The "u" starts a Unicode escape but is not one of its four hex digits.
        context.parameter_value_buffer += char

    elif new_state == DecoderState.PARAMETER_STRING_UNICODE:
        context.unicode_escape_buffer += char
        context.parameter_value_buffer += char

    elif (
        previous_state == DecoderState.PARAMETER_STRING_UNICODE
        and new_state == DecoderState.PARAMETER_STRING_VALUE
    ):
        # Store the fourth hexadecimal digit and finish the Unicode escape.
        context.parameter_value_buffer += char
        context.unicode_escape_buffer = ""

    elif new_state == DecoderState.PARAMETER_STRING_VALUE:
        context.parameter_value_buffer += char
    if new_state == DecoderState.PARAMETER_COMMA:
        context.used_parameter_names.append(context.parameter_name_buffer)
        context.parameter_name_buffer = ""
        context.parameter_value_buffer = ""
        context.current_parameter_type = ""

    if new_state == DecoderState.PARAMETER_COMPLETE:
        if len(context.parameter_name_buffer) > 0:
            context.used_parameter_names.append(context.parameter_name_buffer)
    return context
