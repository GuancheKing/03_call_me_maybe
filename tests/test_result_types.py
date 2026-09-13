from src.generator import build_function_call_result
from src.models import FunctionDefinition, ParameterDefinition


function_definition = FunctionDefinition(
    name="fn_test",
    description="Test function",
    parameters={
        "amount": ParameterDefinition(type="number"),
        "years": ParameterDefinition(type="integer"),
        "name": ParameterDefinition(type="string"),
        "enabled": ParameterDefinition(type="boolean"),
    },
    returns=ParameterDefinition(type="string"),
)


generated_json = """
{
    "name": "fn_test",
    "parameters": {
        "amount": 2,
        "years": 23,
        "name": "Pepo",
        "enabled": true
    }
}
"""


result = build_function_call_result(
    generated_json,
    "Test prompt",
    function_definition,
)


assert result.parameters["amount"] == 2.0
assert isinstance(result.parameters["amount"], float)

assert result.parameters["years"] == 23
assert isinstance(result.parameters["years"], int)

assert result.parameters["name"] == "Pepo"
assert isinstance(result.parameters["name"], str)

assert result.parameters["enabled"] is True
assert isinstance(result.parameters["enabled"], bool)


print("Function call result type normalization tests passed.")
