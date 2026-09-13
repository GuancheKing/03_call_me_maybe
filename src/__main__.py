import argparse
import json
import time

from pathlib import Path
from src.parser import load_function_definitions, load_prompts
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.generator import (
    build_llm_prompt,
    build_token_index,
    build_token_texts,
    generate_function_call_two_phase,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate function calls from natural-language prompts."
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions."
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing input prompts."
    )

    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the JSON file where results will be written."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Limits the number of tokens to this number"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    function_definitions = load_function_definitions(
        args.functions_definition
    )
    prompts = load_prompts(
        args.input
    )
    if not prompts:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as file:
            json.dump([], file, indent=4)

        print("No prompts to process.")
        return
    model = Small_LLM_Model()

    sample_llm_prompt = build_llm_prompt(
        prompts[0].prompt,
        function_definitions
    )

    sample_input_ids = model.encode(sample_llm_prompt)
    sample_input_ids = sample_input_ids[0].tolist()

    sample_logits = model.get_logits_from_input_ids(sample_input_ids)

    token_texts = build_token_texts(
        model,
        len(sample_logits)
    )

    token_index = build_token_index(token_texts)

    results = []
    limit = args.limit
    start_time = time.perf_counter()
    for index, prompt in enumerate(prompts):
        print(f"\nGenerating: {prompt.prompt}")
        prompt_start = time.perf_counter()
        function_call = generate_function_call_two_phase(
            model,
            prompt.prompt,
            function_definitions,
            token_texts,
            token_index,
            limit
        )
        prompt_end = time.perf_counter()
        print(
            "Prompt generation time:",
            round(prompt_end - prompt_start, 3),
            "seconds"
        )
        results.append(function_call)

    results_dict = [
        result.model_dump()
        for result in results
    ]
    results_json = json.dumps(results_dict, indent=4)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        file.write(results_json)

    end_time = time.perf_counter()

    print(
        "Total generation time:",
        round(end_time - start_time, 3),
        "seconds"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nYou are THAT kind of person, innit?\n"
              "Execution interrupted by user.")
