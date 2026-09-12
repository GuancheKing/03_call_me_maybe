import json
from src.models import FunctionCallResult, FunctionDefinition
from src.decoder import (
    DecoderContext, mask_invalid_logits,
    DecoderState, update_context
    )
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


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
        logits: Optional precomputed logits to reuse for the current input.

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


def generate_function_call_single_phase(
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
        token_texts: Decoded text representation of the model vocabulary.
        token_index: Mapping from first characters to candidate token IDs.
        initial_logits: Optional precomputed logits for the first generated
            token.

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
    context = DecoderContext(state=DecoderState.START)
    allowed_names = [function.name for function in function_definitions]
    text = ""
    token_counter = 0
    while context.state != DecoderState.COMPLETE and token_counter < limit:
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
        input_ids.append(next_token_id)
        text += next_token_text
        token_counter += 1
        for char in next_token_text:
            update_context(context, char, allowed_names, function_definitions)
    if context.state != DecoderState.COMPLETE:
        raise ValueError(f"Couldn't complete with tokens limited to {limit}")
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


def select_function_name(
    model: Small_LLM_Model,
    user_prompt: str,
    function_definitions: list[FunctionDefinition],
    token_texts: list[str],
    token_index: dict[str, list[int]],
    limit: int
) -> str:
    """
    Select the function name that best matches a user request.

    Args:
        model: Language model used for function selection.
        user_prompt: Natural-language request from the user.
        function_definitions: Available function definitions.
        token_texts: Decoded text representation of the model vocabulary.
        token_index: Mapping from first characters to candidate token IDs.
        limit: Maximum number of tokens allowed for selection.

    Returns:
        The selected function name.
    """
    selection_prompt = build_function_selection_prompt(
        user_prompt,
        function_definitions
    )
    selection_prompt += '\n{"name":"'
    input_ids = model.encode(selection_prompt)
    input_ids = input_ids[0].tolist()
    allowed_names = [
        function.name
        for function in function_definitions
    ]
    selected_name = ""
    token_counter = 0
    while (
        selected_name not in allowed_names
        and token_counter < limit
    ):
        logits = model.get_logits_from_input_ids(input_ids)
        masked_logits = [float("-inf")] * len(logits)
        allowed_start_chars = {
            name[len(selected_name)]
            for name in allowed_names
            if name.startswith(selected_name)
            and len(name) > len(selected_name)
        }
        for char in allowed_start_chars:
            for token_id in token_index.get(char, []):
                token_text = token_texts[token_id]
                candidate = selected_name + token_text

                if any(
                    name.startswith(candidate)
                    for name in allowed_names
                ):
                    masked_logits[token_id] = logits[token_id]
        best_token_id, best_token_text = select_best_token(
            masked_logits,
            token_texts
        )
        input_ids.append(best_token_id)
        selected_name += best_token_text
        token_counter += 1
    if selected_name not in allowed_names:
        raise ValueError(
            f"Couldn't select a function with tokens limited to {limit}"
        )
    return selected_name


def build_function_selection_prompt(
    user_prompt: str,
    function_definitions: list[FunctionDefinition]
) -> str:
    """
    Build the prompt used to select the best matching function.

    Args:
        user_prompt: Natural-language request from the user.
        function_definitions: Available function definitions.

    Returns:
        A prompt containing the available functions, user request,
        and expected function-call format.
    """
    text = "Available functions:\n"
    for function in function_definitions:
        text += format_function_definition(function)
    text += f"Request: {user_prompt}\n"
    text += 'Return a function call in JSON format with'
    ' keys "name" and "parameters".'
    return text


def build_parameter_generation_prompt(
    user_prompt: str,
    function_definition: FunctionDefinition
) -> str:
    """
    Build the prompt used to generate parameters for a selected function.

    Args:
        user_prompt: Natural-language request from the user.
        function_definition: Function selected during the first phase.

    Returns:
        A prompt containing the selected function definition, user request,
        and expected function-call format.
    """
    text = "Selected function:\n"
    text += format_function_definition(function_definition)
    text += f"\nRequest: {user_prompt}\n"
    text += 'Return a function call in JSON'
    ' format with keys "name" and "parameters".'

    return text


def generate_parameters(
    model: Small_LLM_Model,
    user_prompt: str,
    function_definition: FunctionDefinition,
    token_texts: list[str],
    token_index: dict[str, list[int]],
    limit: int
) -> dict[str, str | int | float | bool]:
    """
    Generate parameters for a selected function using constrained decoding.

    Args:
        model: Language model used to generate parameter values.
        user_prompt: Natural-language request from the user.
        function_definition: Function selected during the first phase.
        token_texts: Decoded text representation of the model vocabulary.
        token_index: Mapping from first characters to candidate token IDs.
        limit: Maximum number of tokens allowed for parameter generation.

    Returns:
        Generated parameter values keyed by parameter name.
    """
    parameter_prompt = build_parameter_generation_prompt(
        user_prompt,
        function_definition
    )
    parameter_prompt += (
        f'\n{{"name":"{function_definition.name}",'
        '"parameters":{'
    )
    input_ids = model.encode(parameter_prompt)
    input_ids = input_ids[0].tolist()
    context = DecoderContext(
        state=DecoderState.PARAMETERS_OPEN,
        function_name_buffer=function_definition.name,
        parameter_names=list(function_definition.parameters.keys())
    )
    allowed_names = [function_definition.name]
    active_function_definitions = [function_definition]
    text = "{"
    token_counter = 0
    while (
        context.state != DecoderState.PARAMETER_COMPLETE
        and token_counter < limit
    ):
        next_token_id, next_token_text = generate_next_token(
            model,
            input_ids,
            token_texts,
            token_index,
            context,
            allowed_names,
            active_function_definitions
        )
        input_ids.append(next_token_id)
        token_counter += 1
        for char in next_token_text:
            text += char
            update_context(
                context,
                char,
                allowed_names,
                active_function_definitions
            )
            if context.state == DecoderState.PARAMETER_COMPLETE:
                break
    if context.state != DecoderState.PARAMETER_COMPLETE:
        raise ValueError(
            f"Couldn't complete parameters with tokens limited to {limit}"
        )
    parameters = json.loads(text)
    return parameters


def generate_function_call_two_phase(
    model: Small_LLM_Model,
    user_prompt: str,
    function_definitions: list[FunctionDefinition],
    token_texts: list[str],
    token_index: dict[str, list[int]],
    limit: int
) -> FunctionCallResult:
    """
    Generate a function call using separate function selection and
    parameter generation phases.

    Args:
        model: Language model used for generation.
        user_prompt: Natural-language request from the user.
        function_definitions: Available function definitions.
        token_texts: Decoded text representation of the model vocabulary.
        token_index: Mapping from first characters to candidate token IDs.
        limit: Maximum number of tokens allowed for each generation phase.

    Returns:
        A validated FunctionCallResult containing the original prompt,
        selected function name, and generated parameters.

    Raises:
        ValueError: If the selected function cannot be found.
    """
    selected_name = select_function_name(
        model,
        user_prompt,
        function_definitions,
        token_texts,
        token_index,
        limit
    )
    selected_function = None
    for function in function_definitions:
        if function.name == selected_name:
            selected_function = function
            break
    if selected_function is None:
        raise ValueError(
            f"Selected function '{selected_name}' was not found."
        )
    parameters = generate_parameters(
        model,
        user_prompt,
        selected_function,
        token_texts,
        token_index,
        limit
    )
    return FunctionCallResult(
        prompt=user_prompt,
        name=selected_name,
        parameters=parameters
    )
