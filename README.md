*This project has been created as part of the 42 curriculum by esnavarr*

# Call Me Maybe

Turning natural language into **function calls** with a 0.6B parameter model.

```
"What is the sum of 2 and 3?"  ->  {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

The program never answers the question. It says which function should be
called, and with which arguments.

## Description

A small language model asked politely for JSON produces valid JSON maybe a
third of the time: it adds ``` fences, explanations, or trailing commas. This
project does not ask politely — it never lets the model write the JSON at all.

Two mechanisms do the work:

* **The JSON skeleton is written by the program.** Braces, quotes, colons and
  argument names are emitted directly as tokens. The model is never consulted
  about punctuation.
* **Where a real decision remains, the model chooses from a list.** The
  program builds a small set of candidate options, sets the score of every
  other token to `-infinity`, and lets the model pick among what survives.

The model therefore only ever answers questions like *"which of these five
function names?"* or *"which of these two numbers from the prompt?"*

## Instructions

```bash
make install      # uv sync: creates .venv and installs the dependencies
make run          # reads data/input/, writes data/output/
make lint         # flake8 . + mypy .
make lint-strict  # flake8 . + mypy --strict
make clean        # remove caches
```

The first run downloads `Qwen/Qwen3-0.6B` (~1.5 GB) into the Hugging Face
cache. It runs on CPU; a GPU is used automatically when available. The
`llm_sdk/` package must sit next to `src/`, as provided.

### Example usage

```bash
# default paths
uv run python -m src

# explicit paths
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

Output (`data/output/function_calling_results.json`):

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {"a": 2.0, "b": 3.0}
    },
    {
        "prompt": "Greet john",
        "name": "fn_greet",
        "parameters": {"name": "john"}
    }
]
```

Any failure — missing file, malformed JSON, unreadable definitions — prints a
message to stderr and exits with a non-zero status, so `make run && ...` stops
instead of continuing on a file that was never written.

## Algorithm explanation

The SDK provides one primitive: `get_logits_from_input_ids(ids)`, the score of
every possible next token. Constrained decoding inserts one step before the
token is chosen:

```
context ──▶ model ──▶ logits ──▶ [ mask: everything but the candidates -> -inf ] ──▶ argmax
```

`LLM._apply_mask` builds an array full of `-infinity` and copies in only the
scores of the allowed token ids. Everything else becomes impossible.

### Choosing an option

`LLM.next_option` receives a list of candidates, each already tokenized, and
narrows them one token at a time:

```python
while options:
    allowed = {option[0] for option in options}      # first token of each survivor
    next_token = self.next_token(context, allowed)    # model picks, masked
    options = [option[1:] for option in options
               if option[0] == next_token and len(option) > 1]
```

Each round: collect the possible next tokens, let the model choose, drop every
candidate that disagrees. A real run of `"What is the sum of 2 and 3?"`:

```
choose among ['fn_add_numbers', 'fn_greet', 'fn_reverse_string',
              'fn_get_square_root', 'fn_substitute_string_with_regex']
    1 token allowed  -> 'fn'        (all five names start the same way)
    5 tokens allowed -> '_add'      <- the real decision
    1 token allowed  -> '_numbers'
choose among ['2', '3'] -> '2'
choose among ['2', '3'] -> '3'
```

Five forward passes for the whole call. Out of 151 936 tokens, the widest
choice the model faced was five.

### Two phases

1. **The function name.** The context ends at `{"name": "` and the candidates
   are the catalogue names, with every definition in the system prompt so the
   model can compare them.
2. **The arguments.** Now that the function is known, the system prompt is
   narrowed to that one definition, and `add_args` walks its parameters in
   order — writing `"name": ` itself and asking the model only for the value.

Candidates for a value depend on the declared type:

| Declared type | Candidates | Written as |
| --- | --- | --- |
| has an `enum` | the allowed values, as complete JSON literals | as given |
| `number` / `integer` | prompt fragments matching the JSON number grammar | bare, `.0` added if needed |
| `boolean` | `true`, `false` | bare |
| anything else | fragments of the prompt | between double quotes |

Enum values are offered as complete literals **including the quotes**, which is
what lets the model distinguish `"read"` from `"readonly"`: the closing quote
is the token that ends the shorter value. A parameter can therefore never leave
its declared set.

## Design decisions

* **Pydantic everywhere.** `Encoder`, `LLM`, `Function` and `CallMeMaybe` are
  all `BaseModel`.
* **Options instead of character-level masking.** String arguments are chosen
  from complete phrases taken from the prompt. This stops the model halting
  mid-word on a rare name, and guarantees the value is something the user
  actually wrote.
* **A separate `Encoder`.** Tokenization is independent of inference. A trie
  over the vocabulary makes encoding a single left-to-right pass.
* **A short instruction during argument generation.** Only the chosen
  function's schema stays in context. Attention cost grows with context
  length, so this is the main speed lever.
* **Keyword-based regex resolution.** A 0.6B model asked for a regex produces
  things like `numbers\d+`, so patterns are resolved from prompt keywords
  instead. This is a deliberate trade-off, and the clearest weak point of the
  design — see *Known limitations*.
* **Native tool-call format.** Qwen's documented `<tools>` / `<tool_call>`
  shape is used rather than an invented one, since the model has seen it.
* **Greedy, no sampling.** Every choice is an `argmax`, so runs are
  reproducible.
* **A fallback instead of a crash.** If the generated text somehow will not
  parse, the call falls back to schema-valid defaults for the chosen function
  and prints a warning, rather than aborting the run and leaving a truncated
  output file behind.

## Performance analysis

Measured on CPU (16 threads, float32) with `Qwen/Qwen3-0.6B`:

| Metric | Value |
| --- | --- |
| Valid JSON | 100% of outputs parse |
| Total time, 11 prompts | ~2 min (limit: 5 min) |
| Forward passes, 2-argument call | 5 |
| Fallbacks used | 0 on either set |
| Same input, two runs | byte-identical |

The bottleneck is `get_logits_from_input_ids`: it recomputes attention over
the whole context on every call, and there is no KV cache to reuse. Cost is
therefore roughly *number of decisions × context length*, which is why the
skeleton is written rather than generated, and why the instruction is narrowed
to a single function for phase two.

## Challenges faced

* **Early stopping on uncommon words.** Character-level masking made the model
  stop mid-word on rare names (`"Greet shrek"` → `"shr"`). Solved by moving to
  options-based generation: extract complete phrases and narrow them by token
  prefix until one remains.
* **The model would not write regexes.** Asked to produce one freely it mixed
  the description into the pattern (`numbers\d+`). A keyword table was added,
  but resolving the pattern from it directly meant the model was not consulted
  at all — a heuristic answering the question instead of a heuristic proposing
  options. The table now supplies *candidates* (plus any quoted word from the
  request, plus a generic `\w+`) and the model chooses among them, exactly as
  it does for every other argument.
* **Context length and speed.** Sending every function definition on every
  logit call was slow; phase two now sends one.
* **The chat template.** Without Qwen's `<|im_start|>` / `<tool_call>`
  structure the model had no idea what format to continue, and quality was
  poor. Adding it helped more than any other prompt change.
* **Untyped candidates broke the JSON.** Because candidates came from prompt
  words with no type check, a numeric parameter could be filled with a word —
  `{"a": first}` — which is not JSON, and the exception took the whole output
  file with it. Fixed by filtering numeric candidates against the JSON number
  grammar, quoting everything that is not a number or boolean, and adding the
  parse fallback above.

## Testing strategy

There is no automated unit-test suite; verification is done by running the
program and checking its output:

* **Unseen catalogues** with different types — compound interest, SQL queries,
  file paths with backslashes, templates containing quotes and braces, enums,
  booleans, integers — to check the solution generalises rather than fitting
  the provided examples.
* **Schema validation** of the output: every entry parsed with `json.loads`,
  then checked for exactly the `prompt` / `name` / `parameters` keys, the right
  argument names, the declared types, and enum membership.
* **Edge cases**: prompts with no number for a numeric parameter, empty
  strings, very large numbers, non-ASCII names, prompts matching no function.
* **Error paths**: missing input file, malformed JSON, non-JSON content,
  missing function definitions — each must print a clear message and exit
  non-zero.
* **Determinism**: the same input run twice must produce identical output.
* **Norm**: `make lint` and `make lint-strict` must both pass.

## Known limitations

* The `regex` parameter is resolved from a keyword table, and
  the branch is keyed on the parameter being named `regex`.
* The encoder takes the longest matching token at each position, which is not
  the BPE merge algorithm the model was trained with. Results are usually
  identical on ASCII, not always.
* Special tokens are not atomic: `<|im_start|>` encodes as six ordinary text
  tokens rather than the single id `151644`, so the chat markers are read as
  text.
* Characters absent from the vocabulary as literal characters are skipped, so
  CJK input loses its content.
* Argument values can only come from the prompt; a value that must be inferred
  cannot be produced.

## Resources

* **Hands-On Large Language Models: Language Understanding and Generation** —
  Jay Alammar & Maarten Grootendorst (O'Reilly, 2024). The clearest treatment
  of tokenization, the token-by-token generation loop, and why a model's output
  distribution is something you can intervene on — which is exactly what this
  project does.
* [Qwen3 documentation](https://qwen.readthedocs.io/en/latest/) and
  [Qwen tool calling format](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
* [BPE tokenization explained](https://huggingface.co/learn/nlp-course/en/chapter6/5)
  (Hugging Face NLP course)
* [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
  — the original BPE paper
* [RFC 8259 — The JSON Data Interchange Format](https://www.rfc-editor.org/rfc/rfc8259)
  — the number and string grammars the type rules implement
* [Pydantic documentation](https://docs.pydantic.dev/) and
  [uv documentation](https://docs.astral.sh/uv/)

### AI usage

Claude (Anthropic) was used for:

* **Debugging**: pydantic initialisation patterns (`PrivateAttr`,
  `super().__init__()`), and identifying Qwen3's tool-calling template.
* **Design discussion**: the keyword-mapping approach for regex resolution, and
  the options-based argument extraction that replaced character-level masking.
* **Documentation**: this README.

Not used for: choosing the overall approach, or accepting code without running
it. Every number in the *Performance analysis* section was measured by running
the program.
