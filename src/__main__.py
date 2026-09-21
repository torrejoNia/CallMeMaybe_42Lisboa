from llm_sdk import Small_LLM_Model
from src.encoder import Encoder
from src.llm import LLM
from src.callmemaybe import CallMeMaybe
import json
import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--functions_definition',
        default='data/input/functions_definition.json'
    )
    parser.add_argument(
        '--input',
        default='data/input/function_calling_tests.json'
    )
    parser.add_argument(
        '--output',
        default='data/output/function_calling_results.json'
    )
    return parser.parse_args()


def create_encoder(vocab_path: str) -> Encoder:
    with open(vocab_path, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    return Encoder(tokens)


if __name__ == "__main__":
    try:
        args = parse_args()
        llm_model = Small_LLM_Model()
        encoder = create_encoder(llm_model.get_path_to_vocab_file())
        llm = LLM(llm_model, encoder)
        cmm = CallMeMaybe(llm, args.functions_definition)

        prompts = None
        with open(args.input) as requests:
            prompts = [t['prompt'] for t in json.load(requests)]
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w') as output:
            output.write('[\n')
            for i, p in enumerate(prompts):
                print(f'\n{i}. Processing \'{p}\'...')
                if i < len(prompts) - 1:
                    output.write(cmm.process_func(p) + ',\n')
                else:
                    output.write(cmm.process_func(p) + '\n')
            output.write(']')
        print('Finished.')

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}", file=sys.stderr)
        sys.exit(1)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e.msg} " +
              f"at line {e.lineno} column {e.colno}", file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)
