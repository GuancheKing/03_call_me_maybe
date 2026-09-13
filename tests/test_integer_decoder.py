from src.decoder import (
    DecoderContext,
    DecoderState,
    update_context,
)
from src.models import FunctionDefinition, ParameterDefinition


def build_function(parameter_type: str) -> FunctionDefinition:
    return FunctionDefinition(
        name="fn_test",
        description="Test function",
        parameters={
            "value": ParameterDefinition(type=parameter_type)
        },
        returns=ParameterDefinition(type="string"),
    )


def run_parameter_value(value: str, parameter_type: str) -> DecoderState:
    function_definition = build_function(parameter_type)

    context = DecoderContext(
        state=DecoderState.PARAMETER_COLON,
        function_name_buffer="fn_test",
        parameter_name_buffer="value",
        parameter_names=["value"],
        current_parameter_type=parameter_type,
    )

    for char in value:
        update_context(
            context,
            char,
            ["fn_test"],
            [function_definition],
        )

        if context.state == DecoderState.INVALID:
            return context.state

    update_context(
        context,
        "}",
        ["fn_test"],
        [function_definition],
    )

    return context.state


assert run_parameter_value("23", "integer") == DecoderState.PARAMETER_COMPLETE
assert run_parameter_value("-23", "integer") == DecoderState.PARAMETER_COMPLETE

assert run_parameter_value("23.5", "integer") == DecoderState.INVALID
assert run_parameter_value("-23.5", "integer") == DecoderState.INVALID

assert run_parameter_value("23", "number") == DecoderState.PARAMETER_COMPLETE
assert run_parameter_value("23.5", "number") == DecoderState.PARAMETER_COMPLETE
assert run_parameter_value("-23.5", "number") == (
    DecoderState.PARAMETER_COMPLETE
    )

print("Integer decoder state tests passed.")
