# Riley: Python Style Guide

- Prioritise an easy to remember and intuitive user API and performant code under the hood.
- Follow the PEP8 style guide: https://peps.python.org/pep-0008/
- Format your code so it is readable, use an 80 character line length and put blank lines around logical groups of statements
- Use descriptive variable names, no single letter variables (double letters for iterators in numpy style are ok) single letter variables for indices / iterators are ok.
- Use major function first variable names: e.g. `FieldScalar`, `FieldVector` and `FieldTensor` instead of `ScalarField`, `VectorField` and `TensorField`.
- Type hint everything: e.g. `def add_ints(a: int, b: int) -> int:`. This makes your code easier to understand and you have the possibility of compiling things if you need.
- `pylint` is a slow linter but will help you if you have type hinted everything. `Ruff` is another good option, it is faster but doesn't pick up type hints as well.
- Use guard clauses (if statements) with returns at the top of functions to reduce the number of nested if/else structures.
- Default mutable data types (lists, dicts, objects) to `None` and then set them with an if statement guard clause
- Use `pathlib` and the `Path` class to manage all file io in preference to manual string handling or the `os` module.
- `numpy` and `scipy` are your friend - avoid for/while loops. Push everything you can down into C. Unless you are writing Cython then loops are great!
- Minimise dependencies as much as possible.
- Avoid decorators unless absolutely necessary (`@dataclass`,  `@abstractmethod` and `@staticmethod` are examples that are ok)
- Don't use `@property`. It is normally used to hide complicated variable initialisation behind the `.` notation - just avoid `@property` altogether and just use a `@dataclass` for data only classes.
- No inheritance unless it is a purely abstract interface (python abstract base class `ABC`) - use composition / dependency injection. See this [video](https://www.youtube.com/watch?v=hxGOiiR9ZKg&t=3s) and thie [video](https://www.youtube.com/watch?v=J1f5b4vcxCQ&t=2s).
- Only use one layer of abstraction - don't inherit from multiple interfaces and don't use mix-ins.
- For interfaces (abstract base classes) prefix the name of the class with a capital `I` e.g. `ISensor`
- For enumerations prefix the name with a capital `E` so `EGeneratorType`.
- Only use abstraction/interfaces when if/else or switch has at least 3 implementations and/or becomes annoying.
- Use a mixture of plain functions and classes with methods where and when they make sense.
- Imports requiring many `.`'s are annoying and the user finds the layers hard to remember. Bring everything to the top level so it can be accessed with `pyvale.`
- Setup good defaults for variables where possible so that the user can get started with minimal input.
- Prefer dataclasses (`@dataclass`) to dictionaries as they tell the user what parameters are needed and can have sensible defaults.
- When using dataclasses `def __post_init__():` is useful for setting defaults for mutable data types.
- Use classes with `__slots__ = ("var1","var2",)` as it is more memory efficient, faster and stops member variables being added dynamically. For dataclasses use: `@dataclass(slots=True)`
- Write docstrings when the code is ready for sharing and use autodocstring to help. For `pyvale` we use `numpy` style docstrings.

