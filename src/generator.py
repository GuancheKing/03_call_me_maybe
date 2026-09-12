import json
from src.models import FunctionCallResult, FunctionDefinition
from src.decoder import (
    DecoderContext, mask_invalid_logits,
    DecoderState, update_context
    )
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
        f"Parameters:\n"
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
        A prompt containing the available functions, expected output
        format, and user's request.
    """
    text = "Functions:\n"

    for function in function_definitions:
        text += format_function_definition(function)

    text += '\nOutput JSON with keys "name" and "parameters".\n'
    text += "Request: " + user_prompt

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


def build_token_index(token_texts: list[str]) -> dict[str, list[int]]:
    """
    Build an index that groups token IDs by their first decoded character.

    Args:
        token_texts: Decoded text associated with each token ID.

    Returns:
        A dictionary where each key is the first character of a token
        and each value is a list of token IDs starting with that character.
    """
    token_dict = {}
    for token_id, token_text in enumerate(token_texts):
        if not token_text:
            continue
        first_char = token_text[0]
        if first_char not in token_dict:
            token_dict[first_char] = [token_id]
        else:
            token_dict[first_char].append(token_id)
    return token_dict


def select_best_token(
    masked_logits: list[float],
    token_texts: list[str]
) -> tuple[int, str]:
    """
    Select the token with the highest valid logit score.

    Args:
        masked_logits: Logit scores after invalid tokens have been masked.
        token_texts: Decoded text associated with each token ID.

    Returns:
        A tuple containing the selected token ID and its decoded text.
    """
    if all(logit == float("-inf") for logit in masked_logits):
        raise ValueError("No valid token available for generation.")
    best_token_id = 0

    for index in range(1, len(masked_logits)):
        if masked_logits[index] > masked_logits[best_token_id]:
            best_token_id = index

    best_token_text = token_texts[best_token_id]
    return best_token_id, best_token_text


def generate_next_token(
    model: Small_LLM_Model,
    input_ids: list[int],
    token_texts: list[str],
    token_index: dict[str, list[int]],
    context: DecoderContext,
    allowed_names: list[str],
    function_definitions: list[FunctionDefinition],
    logits: list[float] | None = None
) -> tuple[int, str]:
    """
    Generate the next valid token according to the decoder constraints.

    Args:
        model: Language model used to obtain token logits.
        input_ids: Token IDs representing the current model input.
        token_texts: Decoded text associated with each token ID.
        token_index: Token IDs grouped by their first decoded character.
        context: Current decoder context.
        allowed_names: Function names available for generation.
        function_definitions: Available function definitions.

    Returns:
        A tuple containing the selected token ID and its decoded text.
    """
    if logits is None:
        logits = model.get_logits_from_input_ids(input_ids)
    masked_logits = mask_invalid_logits(
        logits,
        token_texts,
        token_index,
        context,
        allowed_names,
        function_definitions
    )

    best_token_id, best_token_text = select_best_token(
        masked_logits, token_texts)

    return best_token_id, best_token_text


def generate_function_call(
    model: Small_LLM_Model,
    user_prompt: str,
    function_definitions: list[FunctionDefinition],
    limit: int,
    token_texts: list[str],
    token_index: dict[str, list[int]],
    initial_logits: list[float] | None = None
    ) -> FunctionCallResult:
    """
    Generate a valid function call using constrained token generation.

    Args:
        model: Language model used to generate the function call.
        user_prompt: Natural-language request from the user.
        function_definitions: Available function definitions that the model
            can choose from.
        limit: Maximum number of tokens allowed for generation.

    Returns:
        A validated FunctionCallResult containing the original prompt,
        selected function name, and generated parameters.

    Raises:
        ValueError: If generation does not reach a complete valid output
            before the token limit is reached.
    """
    llm_prompt = build_llm_prompt(
        user_prompt,
        function_definitions
    )
    input_ids = model.encode(llm_prompt)
    input_ids = input_ids[0].tolist()
    initial_token_count = len(input_ids)
    context = DecoderContext(state=DecoderState.START)
    allowed_names = [function.name for function in function_definitions]
    text = ""
    token_counter = 0
    while context.state != DecoderState.COMPLETE and token_counter < limit:
        # next_token_id, next_token_text = generate_next_token(
        #     model,
        #     input_ids,
        #     token_texts,
        #     token_index,
        #     context,
        #     allowed_names,
        #     function_definitions
        # )
        try:
            next_token_id, next_token_text = generate_next_token(
                model,
                input_ids,
                token_texts,
                token_index,
                context,
                allowed_names,
                function_definitions,
                initial_logits if token_counter == 0 else None
            )
        except ValueError:
            print("\n--- GENERATION FAILED ---")
            print("Generated text:", repr(text))
            print("State:", context.state)
            print("Expected key:", context.expected_key)
            print("Function buffer:", repr(context.function_name_buffer))
            print("Parameter name:", repr(context.parameter_name_buffer))
            print("Parameter type:", context.current_parameter_type)
            print("Parameter value:", repr(context.parameter_value_buffer))
            print("Used parameters:", context.used_parameter_names)
            raise
        input_ids.append(next_token_id)
        text += next_token_text
        token_counter += 1
        for char in next_token_text:
            update_context(context, char, allowed_names, function_definitions)
    if context.state != DecoderState.COMPLETE:
        raise ValueError(f"Couldn't complete with tokens limited to {limit}")
    print("Input tokens:", initial_token_count)
    print("Generated tokens:", token_counter)
    return build_function_call_result(text, user_prompt)


def build_function_call_result(
    generated_json: str,
    user_prompt: str
) -> FunctionCallResult:
    """
    Build the final function call result from generated JSON.

    Args:
        generated_json: JSON string generated by the constrained decoder.
        user_prompt: Original user prompt associated with the function call.

    Returns:
        A validated FunctionCallResult containing the prompt, function name,
        and generated parameters.
    """
    parsed_json = json.loads(generated_json)
    result = FunctionCallResult(
        prompt=user_prompt,
        name=parsed_json["name"],
        parameters=parsed_json["parameters"]
    )
    return result
