import json
import re
from typing import Any

from pydantic import BaseModel

from src.encoder import Encoder
from src.function import Function
from src.llm import LLM


REGEX_MAPPING = [
    (['vowel', 'vowels'], r'[aeiouAEIOU]'),
    (
        ['consonant', 'consonants'],
        r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]',
    ),
    (['digit', 'digits', 'number', 'numbers'], r'\\d+'),
    (['uppercase', 'upper', 'capital'], r'[A-Z]+'),
    (['lowercase', 'lower'], r'[a-z]+'),
    (['letter', 'letters', 'alphabetic'], r'[a-zA-Z]+'),
    (['space', 'spaces', 'whitespace'], r'\\s+'),
    (['punctuation', 'special'], r'[^\w\s]'),
    (['alphanumeric'], r'\\w+'),
    (['newline', 'newlines'], r'\\n+'),
    (['tab', 'tabs'], r'\\t+'),
]


# A JSON number, as defined by the JSON grammar. Used to keep a numeric
# argument from being filled with a word taken from the prompt, which would
# produce output that is not valid JSON.
NUMBER_RE = re.compile(r'^-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?$')
INTEGER_RE = re.compile(r'^-?(0|[1-9]\d*)$')

# Values used when no candidate of the right type can be found.
DEFAULTS = {
    'number': '0.0',
    'float': '0.0',
    'integer': '0',
    'boolean': 'false',
}


def escape(text: str) -> str:
    """Escapes backslashes and double quotes in the text."""
    return text.replace('\\', '\\\\').replace('"', '\\"')


class CallMeMaybe(BaseModel):
    llm: LLM
    encoder: Encoder
    functions: dict[str, Function]
    t_defintions: list[int]
    t_instruction_prefix: list[int]
    t_instruction_suffix: list[int]

    def __init__(self, llm: LLM, func_definitons: str) -> None:
        encoder = llm.encoder

        functions = {}
        with open(func_definitons, 'r') as f:
            for func in json.load(f):
                functions[func['name']] = Function(func, encoder)

        t_defintions = [t for f in functions.values() for t in f.t_definition]

        t_instruction_prefix = encoder.encode(
            '<|im_start|>system\n'
            'You are provided with function signatures '
            'within <tools></tools> XML tags:\n'
            '<tools>\n')
        t_instruction_suffix = encoder.encode(
            '</tools>\n'
            'For each function call, return a json '
            'object within <tool_call></tool_call> tags:\n'
            '<tool_call>\n'
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            '</tool_call>\n'
            '<|im_end|>\n')

        super().__init__(
            llm=llm,
            encoder=encoder,
            functions=functions,
            t_defintions=t_defintions,
            t_instruction_prefix=t_instruction_prefix,
            t_instruction_suffix=t_instruction_suffix
        )

    def set_tools(self, func: Function | None = None) -> None:
        """Updates the LLM context with function definitions."""

        if func is not None:
            definitions = func.t_definition
        else:
            definitions = self.t_defintions
        new = self.t_instruction_prefix + definitions
        new += self.t_instruction_suffix
        self.llm.set_instruction(new)

    def regex_pattern(self, text: str) -> list[int]:
        """Resolves the regex pattern from prompt keywords."""

        words = {w.strip('\'\".,!?').lower() for w in text.split()}
        for keywords, pattern in REGEX_MAPPING:
            if words & set(keywords):
                return self.encoder.encode(pattern)

        match = re.search(r"['\"](\w+)['\"]", text)
        if match:
            return self.encoder.encode(match.group(1))

        return self.encoder.encode(r'\w+')

    def enum_value(self, tokens: list[int], enum: list[Any]) -> list[int]:
        """Lets the LLM choose one of the values allowed by the schema.

        Each option is the complete JSON literal, quotes included, so the
        result is always one of the declared values and always valid JSON.
        Keeping the closing quote inside the option also lets the model tell
        "read" from "readonly": the quote is what ends the shorter value.
        """

        options = [
            self.encoder.encode(json.dumps(value))
            for value in enum
        ]
        options = [option for option in options if option]
        if not options:
            print('  Warning: no allowed value could be encoded')
            return self.encoder.encode('null')
        return self.llm.next_option(tokens, options)

    def numeric_options(self, text: str, arg_type: str) -> list[list[int]]:
        """Keeps only the prompt fragments that are valid JSON numbers."""

        pattern = INTEGER_RE if arg_type == 'integer' else NUMBER_RE
        options = []
        for option in self.encoder.encode_words_separated(text):
            if option and pattern.match(self.encoder.decode(option).strip()):
                options.append(option)
        return options

    def as_float(self, tokens: list[int]) -> list[int]:
        """Makes sure a number argument is written as a float literal."""

        literal = self.encoder.decode(tokens)
        if not any(char in literal for char in '.eE'):
            return tokens + self.encoder.encode('.0')
        return tokens

    def default_args(self, function: Function) -> dict[str, Any]:
        """Schema valid arguments, used when generation cannot be parsed."""

        values: dict[str, Any] = {
            'number': 0.0, 'float': 0.0, 'integer': 0, 'boolean': False
        }
        return {
            name: function.enums[name][0] if name in function.enums
            else values.get(arg_type, '')
            for name, arg_type in function.params.items()
        }

    def encode_definition(self, function: Function) -> list[int]:
        """Encodes function definition for LLM context."""

        definition = (
            function.t_name
            + self.encoder.encode(': ')
            + function.t_description
            + self.encoder.encode('\nParameters:\n')
        )
        for arg_name in function.param_names:
            definition += self.encoder.encode(f'\n{arg_name}: ')
            definition += function.t_params[arg_name]
        return definition

    def add_args(self,
                 function: Function,
                 tokens: list[int],
                 text: str) -> list[int]:
        """Generates all arguments for a function call."""

        for i, arg_name in enumerate(function.param_names):
            arg_type = function.params[arg_name]

            if i > 0:
                tokens += self.encoder.encode(', ')
            tokens += self.encoder.encode(f'"{arg_name}": ')

            enum = function.enums.get(arg_name)
            if enum is not None:
                tokens += self.enum_value(tokens, enum)
                continue

            if arg_name == 'regex':
                tokens += self.encoder.encode('"')
                tokens += self.regex_pattern(text)
                tokens += self.encoder.encode('"')
                continue

            if arg_type in ('number', 'float', 'integer'):
                options = self.numeric_options(text, arg_type)
            elif arg_type == 'boolean':
                options = [
                    self.encoder.encode('true'),
                    self.encoder.encode('false')
                    ]
            else:
                options = self.encoder.encode_words_separated(text)
            options = [option for option in options if option]

            # Only numbers and booleans are written bare; everything else
            # goes between quotes so the result stays valid JSON.
            quoted = arg_type not in ('number', 'float', 'integer', 'boolean')
            if quoted:
                tokens += self.encoder.encode('"')

            if options:
                next = self.llm.next_option(tokens, options)
            else:
                next = self.encoder.encode(DEFAULTS.get(arg_type, ''))
            if arg_type in ('number', 'float'):
                next = self.as_float(next)
            tokens += next
            if quoted:
                tokens += self.encoder.encode('"')

        tokens += self.encoder.encode('}\n')
        return tokens

    def process_func(self, prompt: str) -> str:
        prompt = escape(prompt)
        text = (
            '<|im_start|>user\n' +
            prompt +
            '\n<|im_end|>\n'
            '<|im_start|>assistant\n'
            '<tool_call>\n'
            '{"name": "'
        )
        tokens = self.encoder.encode(text)
        self.set_tools()
        func_names = [f.t_name for f in self.functions.values()]
        func_name = self.llm.next_option(tokens, func_names)
        function = self.functions[self.encoder.decode(func_name)]
        tokens += function.t_name
        tokens += self.encoder.encode('", "arguments": {')
        self.set_tools(function)
        tokens = self.add_args(function, tokens, prompt)
        tokens += self.encoder.encode('}')

        raw = self.encoder.decode(tokens)
        tool_json = raw[raw.find('{"name":'):]
        print(repr(tool_json))
        try:
            arguments = json.loads(tool_json)['arguments']
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # The call could not be parsed: fall back to schema valid
            # defaults rather than aborting the run and leaving a
            # truncated, unparseable output file behind.
            print(f'  Warning: using default arguments ({e})')
            arguments = self.default_args(function)

        return (
            '\t{\n'
            f'\t\t"prompt": "{prompt}",\n'
            f'\t\t"name": "{function.name}",\n'
            f'\t\t"parameters": {json.dumps(arguments)}\n'
            '\t}'
        )
