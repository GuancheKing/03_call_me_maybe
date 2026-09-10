from .models import FunctionDefinition
from llm_sdk import Small_LLM_Model


def format_function_definition(function: FunctionDefinition) -> str:
    """
    Format a function definition as readable text for the LLM prompt.

    Args:
        function: Validated function definition to format.

    Returns:
        A formatted string containing the function name, description,
        and parameters.
    """
    text = (
        f"\nName: {function.name}\n"
        f"Description: {function.description}\n"
        f"Parameters: \n"
    )
    for parameter_name, parameter_definition in function.parameters.items():
        text += f"- {parameter_name}: {parameter_definition.type}\n"
    return text


def build_llm_prompt(
        user_prompt: str,
        function_definitions: list[FunctionDefinition]
) -> str:
    """
    Build the full prompt sent to the LLM.

    Args:
        user_prompt: Natural-language request from the user.
        function_definitions: Available validated function definitions.

    Returns:
        A prompt containing the available functions, the expected
        output format, and the user's request.
    """
    text = ("Available functions:\n")
    for function in function_definitions:
        text += format_function_definition(function)
    call_text = (
        "\nReturn a function call in JSON format with the following keys:\n"
        "- name\n- parameters\n"
        )
    text += call_text
    text += "\nUser request:\n" + user_prompt
    return text


def build_token_texts(model: Small_LLM_Model, vocab_size: int) -> list[str]:
    """
    Build a list that maps token IDs to their decoded text.

    Args:
        model: Language model used to decode token IDs.
        vocab_size: Total number of token IDs to decode.

    Returns:
        A list where each index corresponds to a token ID and
        contains the associated decoded text.
    """
    token_ids = [[token_id] for token_id in range(vocab_size)]
    token_texts = model.decode(token_ids)
    return token_texts
