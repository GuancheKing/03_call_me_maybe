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


def is_valid_token(
    token_text: str,
    context: DecoderContext,
    allowed_names: list[str],
    function_definitions: list[FunctionDefinition]
) -> bool:
    """
    Check whether a token is valid from the current decoder state.

    Args:
        token_text: Decoded text of the token to validate.
        context: Current decoding context.
        allowed_names: Function names available for generation.
        function_definitions: Available function definitions.

    Returns:
        True if the whole token can be processed without reaching
        the INVALID state, otherwise False.
    """
    if not token_text:
        return False
    all_safe = True
    safe_states = (
        DecoderState.PARAMETER_VALUE_OPEN,
        DecoderState.PARAMETER_STRING_VALUE)
    if context.state in safe_states:
        for char in token_text:
            if char == '"' or char == "\\" or ord(char) < 0x20:
                all_safe = False
                break
        if all_safe:
            return True
    temp_context = copy_decoder_context(context)
    for char in token_text:
        update_context(temp_context, char,
                       allowed_names, function_definitions)
        if temp_context.state == DecoderState.INVALID:
            return False
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


def can_start_token(
    token_text: str,
    context: DecoderContext,
    allowed_names: list[str]
) -> bool:
    """
    Check whether a token can potentially start from the current state.

    This function performs a cheap check using only the first character
    of the token. Full token validation is performed later by
    is_valid_token().

    Args:
        token_text: Decoded text of the token to check.
        context: Current decoding context.
        allowed_names: Function names available for generation.

    Returns:
        True if the token may be valid from the current state,
        otherwise False.
    """
    if not token_text:
        return False

    char = token_text[0]

    if context.state == DecoderState.START:
        return char == "{"

    elif context.state == DecoderState.OBJECT_OPEN:
        return char == '"'

    elif context.state == DecoderState.KEY_OPEN:
        return (
            bool(context.expected_key)
            and char == context.expected_key[0]
        )

    elif context.state == DecoderState.KEY_TEXT:
        if context.key_buffer == context.expected_key:
            return char == '"'

        if len(context.key_buffer) < len(context.expected_key):
            return char == context.expected_key[len(context.key_buffer)]

        return False

    elif context.state == DecoderState.KEY_CLOSE:
        return char == ":"

    elif context.state == DecoderState.COLON:
        if context.expected_key == "name":
            return char == '"'

        if context.expected_key == "parameters":
            return char == "{"

        return False

    elif context.state == DecoderState.VALUE_OPEN:
        candidate = context.function_name_buffer + char

        for name in allowed_names:
            if name.startswith(candidate):
                return True

        return False

    elif context.state == DecoderState.FUNCTION_NAME:
        if (
            char == '"'
            and context.function_name_buffer in allowed_names
        ):
            return True

        candidate = context.function_name_buffer + char

        for name in allowed_names:
            if name.startswith(candidate):
                return True

        return False

    elif context.state == DecoderState.VALUE_CLOSE:
        return char == ","

    elif context.state == DecoderState.COMMA:
        return char == '"'

    elif context.state == DecoderState.PARAMETERS_OPEN:
        if char == '"':
            return True

        if char == "}" and not context.parameter_names:
            return True

        return False

    elif context.state == DecoderState.PARAMETER_KEY_OPEN:
        candidate = context.parameter_name_buffer + char

        for name in context.parameter_names:
            if (
                name not in context.used_parameter_names
                and name.startswith(candidate)
            ):
                return True

        return False

    elif context.state == DecoderState.PARAMETER_NAME:
        available_names = [
            name
            for name in context.parameter_names
            if name not in context.used_parameter_names
        ]

        if (
            char == '"'
            and context.parameter_name_buffer in available_names
        ):
            return True

        candidate = context.parameter_name_buffer + char

        for name in available_names:
            if name.startswith(candidate):
                return True

        return False

    elif context.state == DecoderState.PARAMETER_KEY_CLOSE:
        return char == ":"

    elif context.state == DecoderState.PARAMETER_COLON:
        if context.current_parameter_type == "string":
            return char == '"'

        if context.current_parameter_type == "number":
            return char in "-0123456789"

        if context.current_parameter_type == "boolean":
            return char in ("t", "f")

        return False

    elif context.state == DecoderState.PARAMETER_VALUE_OPEN:
        return True

    elif context.state == DecoderState.PARAMETER_STRING_VALUE:
        return True

    elif context.state == DecoderState.PARAMETER_STRING_ESCAPE:
        return char in '"\\/bfnrtu'

    elif context.state == DecoderState.PARAMETER_STRING_UNICODE:
        return char in "0123456789abcdefABCDEF"

    elif context.state == DecoderState.PARAMETER_NUMBER_VALUE:
        if (
            char in "0123456789"
            and context.parameter_value_buffer not in ("0", "-0")
        ):
            return True

        if (
            char == "."
            and context.parameter_value_buffer != "-"
            and "." not in context.parameter_value_buffer
        ):
            return True

        if (
            char in (",", "}")
            and is_a_valid_number(context.parameter_value_buffer)
        ):
            return True

        return False

    elif context.state == DecoderState.PARAMETER_BOOLEAN_VALUE:
        if (
            context.parameter_value_buffer in ("true", "false")
            and char in (",", "}")
        ):
            return True

        candidate = context.parameter_value_buffer + char

        return (
            "true".startswith(candidate)
            or "false".startswith(candidate)
        )

    elif context.state == DecoderState.PARAMETER_VALUE_CLOSE:
        return char in (",", "}")

    elif context.state == DecoderState.PARAMETER_COMMA:
        return char == '"'

    elif context.state == DecoderState.PARAMETER_COMPLETE:
        return char == "}"

    elif context.state in (
        DecoderState.COMPLETE,
        DecoderState.INVALID,
        DecoderState.OBJECT_CLOSE,
    ):
        return False

    return False


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
        if ord(char) < 0x20:
            return DecoderState.INVALID
        if char == "\\":
            return DecoderState.PARAMETER_STRING_ESCAPE
        if char != '"':
            return DecoderState.PARAMETER_STRING_VALUE
        return DecoderState.PARAMETER_VALUE_CLOSE
    elif current_state == DecoderState.PARAMETER_STRING_VALUE:
        if ord(char) < 0x20:
            return DecoderState.INVALID
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
        completed_parameter_names = (
            context.used_parameter_names
            + [context.parameter_name_buffer]
        )

        if char == ',':
            for name in context.parameter_names:
                if name not in completed_parameter_names:
                    return DecoderState.PARAMETER_COMMA
            return DecoderState.INVALID

        elif char == '}':
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


def mask_invalid_logits(
        logits: list[float],
        token_texts: list[str],
        token_index: dict[str, list[int]],
        context: DecoderContext,
        allowed_names: list[str],
        function_definitions: list[FunctionDefinition]
        ) -> list[float]:
    """
    Mask logits that correspond to invalid tokens.

    Args:
        logits: Scores assigned to each possible token.
        token_texts: Decoded text associated with each token.
        token_index: Token IDs grouped by their first decoded character.
        context: Current decoding context.
        allowed_names: Function names available for generation.
        function_definitions: Available function definitions.

    Returns:
        A copy of the logits with invalid token scores set to negative inf.

    Raises:
        ValueError: If logits and token_texts have different lengths.
    """
    if len(logits) != len(token_texts):
        raise ValueError(
            "logits and token_texts must have the same length."
            )
    masked_logits = [float("-inf")] * len(logits)
    allowed_start_chars = get_allowed_start_chars(
        context, allowed_names, token_index)
    for char in allowed_start_chars:
        for token_id in token_index.get(char, []):
            if is_valid_token(
                token_texts[token_id],
                context,
                allowed_names,
                function_definitions
            ):
                masked_logits[token_id] = logits[token_id]
    return masked_logits


def get_allowed_start_chars(
        context: DecoderContext,
        allowed_names: list[str],
        token_index: dict[str, list[int]]
        ) -> set[str]:
    allowed = set()
    if context.state == DecoderState.START:
        allowed.add("{")
    elif context.state == DecoderState.OBJECT_OPEN:
        allowed.add('"')
    elif context.state == DecoderState.KEY_CLOSE:
        allowed.add(":")
    elif context.state == DecoderState.VALUE_CLOSE:
        allowed.add(",")
    elif context.state == DecoderState.COMMA:
        allowed.add('"')
    elif context.state == DecoderState.PARAMETER_KEY_CLOSE:
        allowed.add(":")
    elif context.state == DecoderState.PARAMETER_COMMA:
        allowed.add('"')
    elif context.state == DecoderState.PARAMETER_COMPLETE:
        allowed.add("}")
    elif context.state == DecoderState.KEY_OPEN:
        if context.expected_key:
            allowed.add(context.expected_key[0])
    elif context.state == DecoderState.KEY_TEXT:
        if context.key_buffer == context.expected_key:
            allowed.add('"')
        elif len(context.key_buffer) < len(context.expected_key):
            allowed.add(
                context.expected_key[len(context.key_buffer)]
            )
    elif context.state == DecoderState.COLON:
        if context.expected_key == "name":
            allowed.add('"')
        elif context.expected_key == "parameters":
            allowed.add("{")
    elif context.state == DecoderState.VALUE_OPEN:
        for name in allowed_names:
            if (
                name.startswith(context.function_name_buffer)
                and len(context.function_name_buffer) < len(name)
            ):
                allowed.add(
                    name[len(context.function_name_buffer)]
                )
    elif context.state == DecoderState.FUNCTION_NAME:
        if context.function_name_buffer in allowed_names:
            allowed.add('"')
        for name in allowed_names:
            if (
                name.startswith(context.function_name_buffer)
                and len(context.function_name_buffer) < len(name)
            ):
                allowed.add(
                    name[len(context.function_name_buffer)]
                )
    elif context.state == DecoderState.PARAMETERS_OPEN:
        if not context.parameter_names:
            allowed.add("}")
        else:
            allowed.add('"')
    elif context.state == DecoderState.PARAMETER_KEY_OPEN:
        available_names = [
            name
            for name in context.parameter_names
            if name not in context.used_parameter_names
        ]
        for name in available_names:
            if (
                name.startswith(context.parameter_name_buffer)
                and len(context.parameter_name_buffer) < len(name)
            ):
                allowed.add(
                    name[len(context.parameter_name_buffer)]
                )
    elif context.state == DecoderState.PARAMETER_NAME:
        available_names = [
            name
            for name in context.parameter_names
            if name not in context.used_parameter_names
        ]
        if context.parameter_name_buffer in available_names:
            allowed.add('"')
        for name in available_names:
            if (
                name.startswith(context.parameter_name_buffer)
                and len(context.parameter_name_buffer) < len(name)
            ):
                allowed.add(
                    name[len(context.parameter_name_buffer)]
                )
    elif context.state == DecoderState.PARAMETER_COLON:
        if context.current_parameter_type == "string":
            allowed.add('"')
        elif context.current_parameter_type == "number":
            for char in "-0123456789":
                allowed.add(char)
        elif context.current_parameter_type == "boolean":
            allowed.add("t")
            allowed.add("f")
    elif context.state == DecoderState.PARAMETER_VALUE_OPEN:
        for char in token_index.keys():
            allowed.add(char)
    elif context.state == DecoderState.PARAMETER_STRING_VALUE:
        for char in token_index.keys():
            allowed.add(char)
    elif context.state == DecoderState.PARAMETER_STRING_ESCAPE:
        for char in '"\\/bfnrtu':
            allowed.add(char)
    elif context.state == DecoderState.PARAMETER_STRING_UNICODE:
        for char in "0123456789abcdefABCDEF":
            allowed.add(char)
    elif context.state == DecoderState.PARAMETER_NUMBER_VALUE:
        if context.parameter_value_buffer not in ("0", "-0"):
            for char in "0123456789":
                allowed.add(char)
        if (
            context.parameter_value_buffer != "-"
            and "." not in context.parameter_value_buffer
        ):
            allowed.add(".")
        if is_a_valid_number(context.parameter_value_buffer):
            allowed.add(",")
            allowed.add("}")
    elif context.state == DecoderState.PARAMETER_BOOLEAN_VALUE:
        if context.parameter_value_buffer in ("true", "false"):
            allowed.add(",")
            allowed.add("}")
        for value in ("true", "false"):
            if (
                value.startswith(context.parameter_value_buffer)
                and len(context.parameter_value_buffer) < len(value)
            ):
                allowed.add(
                    value[len(context.parameter_value_buffer)]
                )
    elif context.state == DecoderState.PARAMETER_VALUE_CLOSE:
        allowed.add(",")
        allowed.add("}")
    return allowed


def copy_decoder_context(context: DecoderContext) -> DecoderContext:
    """
    Create an independent copy of a decoder context.

    Mutable list fields are copied explicitly so changes made while
    validating a candidate token do not affect the original context.

    Args:
        context: Decoder context to copy.

    Returns:
        A new DecoderContext with the same current values.
    """
    new_context = DecoderContext(
        state=context.state,
        expected_key=context.expected_key,
        key_buffer=context.key_buffer,
        function_name_buffer=context.function_name_buffer,
        parameter_name_buffer=context.parameter_name_buffer,
        parameter_names=context.parameter_names.copy(),
        current_parameter_type=context.current_parameter_type,
        parameter_value_buffer=context.parameter_value_buffer,
        used_parameter_names=context.used_parameter_names.copy(),
        unicode_escape_buffer=context.unicode_escape_buffer
    )
    return new_context
