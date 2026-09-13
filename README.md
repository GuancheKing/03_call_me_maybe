*This project has been created as part of the 42 curriculum by @josjimen.*

# call_me_maybe

## Description

`call_me_maybe` is an introduction to function calling with Large Language Models.

The goal of the project is to transform natural-language requests into structured function calls. Given a list of available function definitions and a user prompt, the program must select the appropriate function and generate all required parameters with the correct types.

The project uses the `Qwen/Qwen3-0.6B` language model through the provided `llm_sdk`.

Because small language models cannot reliably produce valid structured output on their own, the generation process uses constrained decoding. At every generation step, invalid tokens are masked before the next token is selected. This ensures that the generated structure follows the expected JSON format and parameter schema.

The current implementation uses a two-phase generation strategy:

1. Select the most appropriate function using the language model.
2. Generate only the parameters required by the selected function using constrained decoding.

This approach significantly reduces generation time compared with generating the entire function call in a single phase.

---

## Instructions

### Installation

The project uses `uv` for dependency and virtual environment management.

Install all dependencies with:

```bash
make install
```

or directly with:

```bash
uv sync
```

### 42 Campus storage note

On some 42 campus machines, the home directory may have very limited disk space.

If `uv` is installed in `~/.local/bin` but `make` cannot find it, temporarily add that directory to the current shell `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

You can verify that `uv` is available with:

```bash
which uv
uv --version
```

If `uv sync`, `make install`, or model loading fails with an error similar to:

```text
No space left on device
```

the `uv` cache, the project virtual environment, and the Hugging Face model cache can be stored in `sgoinfre` instead of the home directory.

First, create the required directories:

```bash
mkdir -p /sgoinfre/students/$USER/uv-cache
mkdir -p /sgoinfre/students/$USER/venvs/call_me_maybe
mkdir -p /sgoinfre/students/$USER/huggingface
```

Configure `uv` and Hugging Face to use `sgoinfre`:

```bash
export UV_CACHE_DIR=/sgoinfre/students/$USER/uv-cache
export HF_HOME=/sgoinfre/students/$USER/huggingface
```

If a previous installation attempt created a local `.venv`, remove it:

```bash
rm -rf .venv
```

Then create a symbolic link so that the project still sees a normal `.venv` directory while the actual virtual environment is stored in `sgoinfre`:

```bash
ln -s /sgoinfre/students/$USER/venvs/call_me_maybe .venv
```

Install the project dependencies normally:

```bash
make install
```

To verify that the virtual environment is correctly linked:

```bash
ls -ld .venv
```

It should point to a path similar to:

```text
.venv -> /sgoinfre/students/<login>/venvs/call_me_maybe
```

The model files downloaded by Hugging Face will now be stored under:

```text
/sgoinfre/students/<login>/huggingface
```

instead of the home directory.

Then run:

```bash
make run
```

These environment variables only change where dependency caches and model files are stored. They do not modify the project behavior, source code, or repository contents.

The `export` commands apply to the current shell session. If a new terminal session is opened, they may need to be set again.


### Running the program

The program can be executed using the default input and output paths:

```bash
uv run python -m src
```

or:

```bash
make run
```

The default files are:

```text
data/input/functions_definition.json
data/input/function_calling_tests.json
data/output/function_calling_results.json
```

Custom files can be provided through command-line arguments:

```bash
uv run python -m src \
    --functions_definition path/to/functions_definition.json \
    --input path/to/function_calling_tests.json \
    --output path/to/results.json
```

The maximum number of tokens allowed during generation can also be changed:

```bash
uv run python -m src --limit 300
```

The default token limit is `500`.

### Makefile

The Makefile provides the following rules:

```text
make install       Install project dependencies.
make run           Run the program.
make debug         Run the program using Python's pdb debugger.
make clean         Remove temporary Python and mypy cache files.
make lint          Run the mandatory flake8 and mypy checks.
make lint-strict   Run flake8 and mypy in strict mode.
```

Both the mandatory lint configuration and the optional strict mypy configuration are supported.

---

## Algorithm

### Constrained Decoding

The project uses autoregressive generation. The model generates one token at a time, and each selected token is appended to the existing sequence before requesting the logits for the next token.

Instead of allowing the model to freely select any token from its vocabulary, the program restricts generation according to the current decoding state.

The decoder maintains a `DecoderContext` containing information such as:

- the current decoder state;
- the function name currently being generated;
- the parameter currently being generated;
- the expected parameter type;
- parameters that have already been generated;
- temporary buffers for function names, parameter names and values;
- string escape and Unicode escape information.

For every generation step:

1. The language model produces logits for the next token.
2. The decoder determines which token prefixes are possible from the current state.
3. Only relevant candidate tokens are inspected.
4. Each candidate token is validated character by character against the decoder state machine.
5. Invalid tokens are assigned negative infinity.
6. The highest-scoring valid token is selected.
7. The decoder context is updated with the generated token.
8. Generation continues until the required structure is complete.

This prevents the language model from producing tokens that would break the expected JSON structure or violate the parameter schema.

### Two-Phase Generation

The first implementation generated the complete function call in a single decoding process.

Although this approach worked, every prompt contained all available function definitions throughout the complete generation process. This made generation significantly slower.

The final implementation therefore separates generation into two phases.

#### Phase 1: Function selection

The model receives the user request together with all available function definitions.

Constrained decoding limits the generated function name to one of the available function names.

The function is still selected by the language model. No keyword matching or heuristic function-selection rules are used.

#### Phase 2: Parameter generation

Once a function has been selected, a new prompt is created containing only:

- the original user request;
- the selected function;
- its description;
- its parameter definitions.

The decoder then generates only the parameters required by that function.

This considerably reduces the amount of context processed while generating parameter values and improves execution time.

### Decoder State Machine

The constrained decoder is implemented as a finite-state machine.

Some of the main states include:

```text
START
OBJECT_OPEN
KEY_OPEN
KEY_TEXT
FUNCTION_NAME
PARAMETERS_OPEN
PARAMETER_NAME
PARAMETER_COLON
PARAMETER_STRING_VALUE
PARAMETER_STRING_ESCAPE
PARAMETER_STRING_UNICODE
PARAMETER_NUMBER_VALUE
PARAMETER_BOOLEAN_VALUE
PARAMETER_COMPLETE
COMPLETE
INVALID
```

Each generated character causes a transition from the current state to another state.

For example, when generating a parameter declared as a string, the decoder requires an opening quotation mark and then enters the string-generation states.

For numeric parameters, quotation marks are not permitted and the generated value is validated as either a `number` or an `integer`.

If a generated character is not valid for the current state, the candidate token is rejected before it can be selected.

The decoder also keeps track of required parameter names so that:

- only parameters defined by the selected function can be generated;
- duplicate parameters are rejected;
- all required parameters must be present before the parameter object can be closed;
- functions with no parameters can immediately produce an empty parameter object.

### Type Validation

The implementation currently supports the parameter types required by the project evaluation:

- `string`
- `number`
- `integer`
- `boolean`

Strings support common JSON escape sequences and Unicode escape syntax.

Numeric values are validated before the parameter object can be closed.

`number` and `integer` are handled separately:

```text
number   -> integer or decimal JSON number
integer  -> integer JSON number only
```

When the final result is constructed, schema `number` values are normalized to Python `float` values and schema `integer` values remain Python `int` values.

Boolean parameters are restricted to valid JSON values:

```text
true
false
```

String parameters also include a conservative literal-preservation step. When the model-generated value uniquely matches text from the original request ignoring capitalization, the exact spelling and capitalization from the user request are preserved.

---

## Design Decisions

### Pydantic models

Pydantic is used to validate the main project data structures.

Models are used for:

- function definitions;
- parameter definitions;
- input prompts;
- final function-call results;
- decoder context.

This provides explicit structures and type validation while keeping parsing logic separate from generation logic.

### Character-level validation of tokens

The model generates tokens, but a single token may contain several characters.

For this reason, candidate tokens are validated character by character against a temporary copy of the decoder context.

A token is accepted only when every character in that token produces a valid state transition.

### Token indexing

Validating every token in the full vocabulary at every generation step was unnecessarily expensive.

To reduce this cost, decoded vocabulary tokens are indexed by their first character.

The decoder first determines which starting characters are possible from the current state and then validates only tokens belonging to those groups.

This reduces the number of candidate tokens that require full validation.

### Two-phase generation

The two-phase implementation was tested on a 42 campus machine using the full 11-prompt benchmark.

The complete generation process took approximately 80 seconds in that environment.

An earlier single-phase implementation required approximately 9 minutes for the same benchmark during development.

The two-phase architecture therefore reduced generation time significantly while preserving high semantic accuracy.

Execution time may still vary depending on the machine, available resources, and whether the model has already been downloaded and cached.


### No heuristic function selection

Function selection remains model-driven.

The implementation does not inspect keywords in the user request to manually select a function. Constrained decoding only limits the model to valid function names.

### Separation of parsing and CLI error handling

Parsing functions validate input files and raise exceptions when the input is invalid.

User-facing error handling is performed at the CLI level instead of printing errors inside lower-level parsing functions.

This avoids duplicated error messages and allows malformed files, missing files and invalid arguments to terminate cleanly without unexpected tracebacks.

---

## Performance Analysis

The implementation was evaluated using a set of representative function-calling scenarios covering both common and edge cases.

The tested cases included:

- function selection from multiple available definitions;
- `number`, `integer`, `string`, and `boolean` parameters;
- mixed `number` and `integer` parameters;
- regular expression arguments;
- SQL queries;
- Linux file paths;
- Windows paths with escaped backslashes;
- template strings containing quotes and placeholders;
- large numeric values;
- functions with no parameters;
- malformed or invalid input files.

Across the main benchmark of 11 representative prompts, the final implementation produced correct results for all tested cases.

A second set of more demanding scenarios achieved 9 correct results out of 11. The remaining failures were related to exact string extraction rather than JSON structure or function selection.

The two-phase implementation completes the 11-prompt benchmark in approximately 3 to 4 minutes in the development environment.

An earlier single-phase implementation required approximately 9 minutes for the same benchmark.

The two-phase architecture therefore reduced generation time by more than half while preserving high semantic accuracy.

Actual execution time depends on the machine and environment used to run the model.

Structural reliability is provided by constrained decoding: invalid JSON and schema transitions are prevented during generation instead of being repaired afterwards.

---

## Challenges Faced

### Generation performance

The initial constrained decoder inspected a large number of vocabulary tokens repeatedly, making generation too slow.

The first major optimization was grouping tokens by their first decoded character so that only potentially valid candidates were fully inspected.

A second major improvement was separating function selection from parameter generation. This reduced the amount of context required during parameter generation and brought execution below the project time limit.

### Balancing prompt detail and speed

Reducing function descriptions too aggressively improved speed but negatively affected semantic function selection.

The final implementation keeps enough information about each function for reliable selection while using the two-phase architecture to reduce the cost during parameter generation.

### Numeric types

The original decoder treated all numeric values as the same type.

Testing revealed the need to correctly distinguish between JSON `number` and `integer` parameters.

Separate validation rules were introduced while reusing the same numeric decoding state.

Final values are also normalized according to the function schema so that `number` parameters become Python floats and `integer` parameters remain integers.

### Preserving literal string values

Small language models may modify capitalization or formatting while extracting values.

For example, a lowercase value present in a request may be interpreted as a proper name and capitalized by the model.

A conservative literal-preservation step was introduced. When a generated string uniquely corresponds to text already present in the original request, the original spelling and capitalization are restored.

### Error handling

Malformed JSON files, missing files, empty function definitions and invalid token limits can otherwise cause confusing runtime failures.

Input parsing and CLI error handling were separated so these cases now terminate with clear error messages instead of unexpected tracebacks.

---

## Testing Strategy

The implementation was evaluated using a representative set of function-calling scenarios covering both common and edge cases.

The tested cases included:

- function selection from multiple available definitions;
- `number`, `integer`, `string`, and `boolean` parameters;
- mixed `number` and `integer` parameters;
- regular expression arguments;
- SQL queries;
- Linux file paths;
- Windows paths with escaped backslashes;
- template strings containing quotes and placeholders;
- large numeric values;
- functions with no parameters;
- malformed or invalid input files.

Across the main benchmark of 11 representative prompts, the final implementation produced correct results for all tested cases.

A second set of more demanding scenarios produced correct results in 9 out of 11 cases. The remaining failures were related to exact string extraction rather than JSON structure, type handling, or function selection.

The two-phase implementation completes the 11-prompt benchmark in approximately 3 to 4 minutes in the development environment.

An earlier single-phase implementation required approximately 9 minutes for the same benchmark.

The two-phase architecture therefore reduced generation time by more than half while preserving high semantic accuracy.

Actual execution time depends on the machine and environment used to run the model.

Structural reliability is provided by constrained decoding: invalid JSON and schema transitions are prevented during generation instead of being repaired afterwards.

---

## Example Usage

Run the program using the default files:

```bash
uv run python -m src
```

Example input prompt:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?"
    }
]
```

Example function definition:

```json
{
    "name": "fn_add_numbers",
    "description": "Add two numbers.",
    "parameters": {
        "a": {
            "type": "number"
        },
        "b": {
            "type": "number"
        }
    },
    "returns": {
        "type": "number"
    }
}
```

Example generated result:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0
        }
    }
]
```

Run the program with custom files:

```bash
uv run python -m src \
    --functions_definition custom/functions.json \
    --input custom/prompts.json \
    --output custom/results.json
```

Run using the Makefile:

```bash
make run
```

Run using Python's debugger:

```bash
make debug
```

Run the mandatory static checks:

```bash
make lint
```

Run the optional strict static checks:

```bash
make lint-strict
```

---

## Resources

The following resources were used to understand the technologies and concepts involved in the project:

- Python documentation: https://docs.python.org/3/
- Python `json` module: https://docs.python.org/3/library/json.html
- Python `enum` module: https://docs.python.org/3/library/enum.html
- Python `argparse` module: https://docs.python.org/3/library/argparse.html
- Pydantic documentation: https://docs.pydantic.dev/
- mypy documentation: https://mypy.readthedocs.io/
- Flake8 documentation: https://flake8.pycqa.org/
- uv documentation: https://docs.astral.sh/uv/
- JSON specification: https://www.json.org/
- Finite-state machine overview: https://en.wikipedia.org/wiki/Finite-state_machine
- Qwen documentation and model information: https://qwenlm.github.io/
- The project subject and the provided `llm_sdk`.

### AI Usage

AI tools were used selectively as a support resource during the project.

They were mainly used for:

- clarifying specific concepts related to constrained decoding, finite-state machines, token generation, and Python typing;
- discussing alternative implementation approaches;
- generating additional test cases;
- reviewing isolated code sections during debugging;
- helping with documentation, docstrings, and wording.

All final code, architecture and implementation decisions were developed by the author, with AI used only to clarify concepts, review ideas and support the learning process.