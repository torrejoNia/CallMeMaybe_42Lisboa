# How this project works, in order

A walkthrough of what happens between typing `make run` and getting
`data/output/function_calling_results.json`, and why each file exists.

Read it top to bottom once; after that use the [quick reference](#quick-reference)
at the end.

---

## 1. The one idea

The program must turn a sentence into a function call:

```
"What is the sum of 2 and 3?"  ->  {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

A language model does not produce JSON. At each step it produces a **score for
every token it knows** — 151 936 numbers — and something has to pick one. Asked
politely for JSON, a 0.6B model will happily answer with ```` ``` ````, or
`Sure!`, or a trailing comma.

This project never lets it get that far. The approach has two halves:

1. **The JSON skeleton is written by the program, not by the model.**
   `{"name": "`, `", "arguments": {`, `"a": `, `}` — all of it is emitted
   directly as tokens. The model is never consulted about punctuation.
2. **Where a real decision remains, the model chooses from a list.**
   The program builds a small set of *candidate options*, masks every other
   token's score to `-infinity`, and lets the model pick among what is left.

```
context -> model -> 151936 scores -> [ everything but the candidates -> -inf ] -> argmax
```

So the model answers questions like *"which of these five function names?"* or
*"which of these two numbers from the prompt?"* — never *"please write me some
JSON."*

---

## 2. Map of the files

In the order the program touches them:

| File | Lines | Job |
| --- | --- | --- |
| [src/__main__.py](src/__main__.py) | 73 | arguments, wiring, the prompt loop, error handling |
| [src/encoder.py](src/encoder.py) | 113 | text ↔ token ids, and splitting a prompt into candidate fragments |
| [src/llm.py](src/llm.py) | 83 | the only file that talks to the model: masking and choosing |
| [src/function.py](src/function.py) | 94 | one callable function: its name, types, enums, tokenized schema |
| [src/callmemaybe.py](src/callmemaybe.py) | 272 | the pipeline: prompt, function choice, argument generation |

Roughly 640 lines total. Every class is a pydantic `BaseModel`, which the
subject asks for explicitly.

---

## 3. The run, step by step

### Step 0 — `uv run python -m src`

[src/__main__.py](src/__main__.py) is both the entry point and the wiring.

### Step 1 — arguments

`parse_args()` at [src/__main__.py:11](src/__main__.py#L11) declares the three
arguments the subject requires — `--functions_definition`, `--input`,
`--output` — with the required defaults.

### Step 2 — load the model and build the encoder

```python
llm_model = Small_LLM_Model()
encoder = create_encoder(llm_model.get_path_to_vocab_file())
```

`get_path_to_vocab_file()` returns the path to the model's `vocab.json`; the
encoder is built from that file alone. Only two SDK methods are used in the
whole project — this one and `get_logits_from_input_ids` — and no private
attribute is ever touched.

### Step 3 — the encoder

[src/encoder.py](src/encoder.py) does two different jobs, and it helps to keep
them apart.

**Job one: text to token ids.** `__init__` builds a **trie** (a character tree)
out of the whole vocabulary, so that `encode()` at
[src/encoder.py:32](src/encoder.py#L32) can walk the text and take the longest
token that matches at each position. Spaces, tabs and newlines are swapped for
the `Ġ`, `Ċ`, `ĉ` characters the vocabulary uses, by `standart_to_special()`.

**Job two: cutting a prompt into candidate values.**
`encode_words_separated()` at [src/encoder.py:71](src/encoder.py#L71) is the
one that matters for arguments. It returns the prompt chopped into pieces, each
already tokenized:

* everything after a colon, as a single piece (so `Format template: Say "hi"`
  yields the whole `Say "hi"`),
* then every "word", where `WORD_PATTERN` treats a quoted span as one word —
  `'SELECT * FROM users'` stays in one piece — and surrounding quotes and
  punctuation are stripped.

Those pieces become the options the model chooses between.

### Step 4 — the model wrapper

[src/llm.py](src/llm.py) is small and is where the constraint is actually
applied.

* `get_logits()` calls the SDK, prepending a stored *instruction* (the system
  prompt with the tool definitions) to the context.
* `_apply_mask()` at [src/llm.py:69](src/llm.py#L69) is the whole trick: build
  an array full of `-infinity`, copy in only the scores of the allowed token
  ids, and return it. Everything else is now impossible.
* `next_token()` takes the `argmax` of that masked array.
* `next_option()` at [src/llm.py:29](src/llm.py#L29) is the important one — see
  the next section.

### Step 5 — the function catalogue

`CallMeMaybe.__init__` at [src/callmemaybe.py:58](src/callmemaybe.py#L58) reads
`functions_definition.json` and builds one [Function](src/function.py) per
entry. Each `Function` stores its name, description, parameter types, any
**enum** restrictions, and — pre-tokenized once, because it is re-sent on every
prompt — its JSON tool schema.

It also builds the system prompt around them, in Qwen's native tool-calling
shape:

```
<|im_start|>system
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "fn_add_numbers", "description": ..., "parameters": {...}}
</tools>
For each function call, return a json object within <tool_call></tool_call> tags:
...
<|im_end|>
```

### Step 6 — one prompt at a time

`process_func()` at [src/callmemaybe.py:233](src/callmemaybe.py#L233) runs the
two phases described below, then parses the result and formats one output
entry. The loop in `__main__` writes them into the JSON array.

### Step 7 — errors

Every failure is caught at [src/__main__.py:58](src/__main__.py#L58), printed
to **stderr**, and exits **non-zero** (1, or 130 for Ctrl-C). The exit code
matters: a grader that runs `make run && make grade` uses that number, not the
message, to decide whether the run worked.

---

## 4. How the constraint actually works

`next_option(tokens, options)` is the heart of the project. It is given a list
of candidate options, each already tokenized, and returns the one the model
likes best — one token at a time:

```python
while options:
    allowed = {option[0] for option in options}   # first token of each survivor
    next_token = self.next_token(context, allowed)  # model picks, masked
    result.append(next_token)
    options = [option[1:] for option in options
               if option[0] == next_token and len(option) > 1]
```

Each round: collect the possible next tokens, let the model choose one, then
drop every option that disagrees. When no option has anything left, the choice
is complete.

### A real trace

Running `"What is the sum of 2 and 3?"` against the provided catalogue:

```
choose among ['fn_add_numbers', 'fn_greet', 'fn_reverse_string',
              'fn_get_square_root', 'fn_substitute_string_with_regex']
    1 token allowed  -> 'fn'          (all five names start the same way)
    5 tokens allowed -> '_add'        <- the real decision
    1 token allowed  -> '_numbers'
  => 'fn_add_numbers'

choose among ['2', '3']
    2 tokens allowed -> '2'
  => '2'

choose among ['2', '3']
    2 tokens allowed -> '3'
  => '3'

'{"name": "fn_add_numbers", "arguments": {"a": 2.0, "b": 3.0}\n}'
```

**Five forward passes for the whole prompt.** Out of 151 936 tokens, the
widest choice the model ever faced was five. Everything else — the braces, the
quotes, the argument names `"a"` and `"b"`, the `.0` that makes it a float —
was written by the program.

### The two phases

Look at `process_func()` and you can see the split:

1. **Choose the function.** The context ends at `{"name": "` and the options
   are the catalogue names. `set_tools()` puts *all* the definitions in the
   system prompt so the model can compare them.
2. **Fill the arguments.** Now that the function is known, `set_tools(function)`
   narrows the system prompt to **that one definition**, and
   `add_args()` at [src/callmemaybe.py:179](src/callmemaybe.py#L179) walks its
   parameters in order, writing `"name": ` itself and asking the model only for
   the value.

Phase two is where the type rules live:

| Declared type | Candidate options | Written as |
| --- | --- | --- |
| has an `enum` | the allowed values, as complete JSON literals | as given |
| `number` / `integer` | prompt fragments that **match the JSON number grammar** | bare, `.0` added if needed |
| `boolean` | `true`, `false` | bare |
| anything else | prompt fragments | between quotes |

The numeric filter (`numeric_options`, [src/callmemaybe.py:135](src/callmemaybe.py#L135))
is what keeps the output parseable. Without it, a prompt with no digits in it
would let the model answer `first`, producing `{"a": first}` — which is not
JSON, and would take the whole output file down with it.

---

## 5. Enum support

When a parameter declares a closed set:

```json
"mode": {"type": "string", "enum": ["read", "readonly", "write"]}
```

`Function` keeps the list and advertises it in the tool schema, so the model
sees the allowed values. `enum_value()` at
[src/callmemaybe.py:116](src/callmemaybe.py#L116) then offers each allowed
value as a **complete JSON literal, quotes included**, and lets `next_option`
choose:

```
choose among ['"read"', '"readonly"', '"write"']
    2 tokens allowed -> '"'
    2 tokens allowed -> 'readonly'
    1 token allowed  -> '"'
  => '"readonly"'
```

Including the quotes is not cosmetic. `next_option` drops an option as soon as
it runs out of tokens, so with bare `read` / `readonly` the shorter value could
never win — once `read` was matched it would be discarded and `readonly` would
be forced. The closing quote is what lets the model *end* the shorter value:
after `read`, the survivors offer `"` and `only`, and the model picks.

The result is that a parameter can no longer leave its declared set. Asked for
a firmware channel that does not exist, the program returns a legal one rather
than inventing `"experimental"`.

---

## 6. What is chosen, and what is written

This is the distinction to be clear about, because it is where the design's
strengths and its limits both come from.

**The model chooses:** which function, and which candidate fills each
argument.

**The program writes:** every brace, quote, colon and comma; every argument
name; the `.0` on whole numbers; and — for one specific parameter name — the
regex.

That last one deserves its own paragraph. `REGEX_MAPPING` at
[src/callmemaybe.py:12](src/callmemaybe.py#L12) is a lookup table from keywords
to patterns (`vowels` → `[aeiouAEIOU]`, `numbers` → `\\d+`, ...), and
`add_args` applies it whenever a parameter is literally named `regex`. No model
call is involved. It produces good answers on the provided tests, but it is a
hardcoded heuristic keyed on a parameter name, and the subject warns against
hardcoding to the provided examples. Expect to be asked about it.

---

## 7. Design decisions

**Options instead of a grammar.** The model is only ever asked to choose
between whole, pre-built candidates. That makes each decision cheap — five
forward passes for a two-argument call — and it means argument values are
copied from the prompt rather than invented, which is usually what you want for
function calling.

The cost is that the constraint is a *list*, not a *structure*. Nothing checks
that the emitted document is valid JSON; validity comes from the skeleton being
written by hand and from the type rules in `add_args`. Where those rules do not
reach, validity is not guaranteed by construction.

**Tool-call format.** The system prompt uses Qwen's native `<tools>` /
`<tool_call>` shape rather than an invented one, because the model has seen it
in training and gives better answers inside it.

**Re-narrowing the tools for phase two.** Once the function is known, the
system prompt is rebuilt with only that function. The model has less to read
and cannot be distracted by other signatures.

**Greedy, no sampling.** Every choice is an `argmax`, so two runs of the same
input give byte-identical output — verified.

**A fallback instead of a crash.** If the generated text somehow will not
parse, `process_func` falls back to schema-valid defaults for the chosen
function and prints a warning, rather than letting the exception unwind and
leave a truncated output file. On the provided tests it never triggers.

---

## 8. Known limitations

Worth knowing before someone else finds them.

**The encoder is not a BPE tokenizer.** It takes the longest matching token at
each position, which is not the same algorithm the model was trained with. On
ASCII the result is usually identical; on `"Greet shrek"` it is not
(`['Gre','et','Ġshr','ek']` instead of `['G','reet','Ġsh','rek']`).

**Special tokens are not atomic.** `<|im_start|>` encodes to six ordinary text
tokens instead of the single id `151644`. The model reads the chat markers as
text rather than as control tokens. It still works — the text is meaningful —
but the prompt is not quite the format the README describes.

**Non-ASCII is unreliable.** Characters that are not in the vocabulary as
literal characters are skipped by `encode()`. Accented Latin usually survives;
CJK does not, so `"Greet 王小明"` loses the name and the argument falls back to
another word from the prompt.

**Values can only come from the prompt.** An argument the request implies but
does not spell out cannot be produced. `"Read the file at /home/user/data.json"`
works, but a value requiring inference does not.

**Nothing stops a value being reused.** In the trace above, the second
argument's options are still `['2', '3']` — the first choice is not removed.
Two parameters can end up with the same value.

**Prefix collisions for bare literals.** The quote trick fixes this for
strings, but a numeric enum of `[1, 12]` cannot resolve to `1`, because there
is no terminator token to end the shorter option.

**Greedy option choice.** `next_option` commits to the highest-scoring *first
token* and cannot reconsider. If the best first token belongs to the wrong
option, that option wins. Scoring each complete candidate instead would fix it,
at several times the forward passes.

---

## 9. Loose ends

Dead code, found while writing this:

* `Encoder.encode_words()` ([src/encoder.py:56](src/encoder.py#L56)) is never
  called — `encode_words_separated()` replaced it.
* `CallMeMaybe.encode_definition()` ([src/callmemaybe.py:165](src/callmemaybe.py#L165))
  is never called; `set_tools()` uses the pre-tokenized `t_definition` instead.
  It is also the only user of `Function.t_params` and `Function.t_description`,
  so those two properties are dead with it.

Also: the `moulinette/` directory is committed to the repository. It is a
downloadable attachment on the evaluation sheet, and it contains the private
test set together with the expected answers. It should not ship — and while it
is there, `make lint` fails on it unless both linters are told to skip it,
because the grader's own code does not follow the norm.

---

## Quick reference

**"Where is X?"**

| Question | File |
| --- | --- |
| Where are illegal tokens removed? | `_apply_mask` in [src/llm.py:69](src/llm.py#L69) |
| How is one option chosen? | `next_option` in [src/llm.py:29](src/llm.py#L29) |
| Where is the model actually called? | `get_logits` in [src/llm.py:56](src/llm.py#L56) |
| How is the function chosen? | phase 1 of `process_func` in [src/callmemaybe.py:233](src/callmemaybe.py#L233) |
| How is an argument kept to its type? | `add_args` in [src/callmemaybe.py:179](src/callmemaybe.py#L179) |
| Where do candidate values come from? | `encode_words_separated` in [src/encoder.py:71](src/encoder.py#L71) |
| How are enums enforced? | `enum_value` in [src/callmemaybe.py:116](src/callmemaybe.py#L116) |
| What happens if a file is broken? | the handlers at [src/__main__.py:58](src/__main__.py#L58) |
| Where would I add a new parameter type? | the type ladder in `add_args` |

**Commands**

```bash
make run          # data/input/ -> data/output/
make lint         # flake8 . + mypy .
make lint-strict  # flake8 . + mypy --strict

# grade against the hidden set
cd moulinette
uv run python -m moulinette prepare_exercises --set private
uv run python -m moulinette grade_student_answers --set private \
    --student_answer_path ../data/output/function_calling_results.json
```
