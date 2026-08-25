# Riley: Python Style Guide

## General Coding Guidance
- Prioritise an easy to remember and intuitive user API and performant code under the hood.
- Follow the PEP8 style guide: https://peps.python.org/pep-0008/
- Format your code so it is readable, use an 80 character line length and put blank lines around logical groups of statements
- Use descriptive variable names, no single letter variables (double letters for iterators in numpy style are ok) single letter variables for indices / iterators are ok.
- Abbreviations are ok in variable names as long as they are not ambiguous for examples `calc` for `calculate`.
- Functions should have a verb as the first word in the function name that indicates what the function actually does.
- Avoid using magic numbers in code. If you need to use magic numbers, make
  them a named module constant with a descriptive name and add a comment when
  the name is not self-explanatory.
- Keep comprehensions to one line with one `for` loop and at most one function
  call. Split comprehensions containing filters, nested loops, nested
  comprehensions or multiple function calls into explicit statements and
  loops.
- Keep `if` conditions to at most two lines and avoid nested function calls in
  conditions. Calculate complex predicates in clearly named intermediate
  statements before the `if`.
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

## Function Verb Meanings

- `apply`: apply an already calculated operation, permutation or mask to data.
- `build`: assemble and return a compound structure, such as topology or an
  adjacency map.
- `calc`: derive and return a new numeric value, array, mask or permutation.
- `check`: test a condition and return a boolean result without modifying the
  input. Use `validate` instead when invalid input raises an exception.
- `convert`: change the representation, data type or shape of a value and
  return the converted value.
- `copy`: return a new copy of an object, optionally replacing selected data.
- `enforce`: return data transformed to satisfy Riley's standard convention.
- `extract`: select and return a meaningful subset of existing data.
- `find`: search for and return a value whose location or existence is not
  already known.
- `get`: retrieve or cheaply look up existing data or metadata.
- `infer`: determine semantic information from geometry, connectivity or other
  evidence where the answer is not stored explicitly.
- `load`: read data from an external source and return its in-memory form.
- `match`: associate candidates with target roles according to stated rules.
- `normalise`: return an equivalent value in a standard representation or
  numerical range.
- `order`: determine or apply a meaningful sequence to existing values.
- `prepare`: perform the named prerequisite transformations for a subsequent
  operation and return the prepared data.
- `process`: avoid this verb when a more precise verb describes the operation;
  use it only for a genuine multi-stage pipeline.
- `restore`: transform standard or working data back to its source
  representation.
- `reverse`: return values with the relevant ordering or orientation reversed.
- `save`: write in-memory data to an external destination.
- `update`: modify an object's stored state in place.
- `validate`: verify input requirements and raise a clear exception when they
  are not satisfied.

## Abbreviations

- array, Array -> arr, Arr
- boolean, Boolean -> bool, Bool
- calculate, Calculate -> calc, Calc
- component, Component -> comp, Comp
- configuration, Configuration -> config, Config
- connectivity, Connectivity -> connect, Connect
- coordinate, Coordinate -> coord, Coord
- convert, Convert -> conv, Conv
- destination, Destination -> dest, Dest
- dimension, Dimension -> dim, Dim
- displacement, Displacement -> disp, Disp
- direction, Direction -> direct, Direct
- element, Element -> elem, Elem
- error, Error -> err, Err
- equivalent, Equivalent -> equiv, Equiv
- geometry, Geometry -> geom, Geom
- global, Global -> glob, Glob
- identifier, Identifier -> id, Id
- image, Image -> img, Img
- independent, Independent -> indep, Indep
- index, Index -> idx, Idx
- indices, Indices -> idxs, Idxs
- local, Local -> loc, Loc
- maximum, Maximum -> max, Max
- minimum, Minimum -> min, Min
- number, Number -> num, Num
- orientation, Orientation -> orient, Orient
- parameter, Parameter -> param, Param
- permutation, Permutation -> perm, Perm
- pixel, Pixel -> px, Px
- projection, Projection -> proj, Proj
- reference, Reference -> ref, Ref
- relative, Relative -> rel, Rel
- simulation, Simulation -> sim, Sim
- source, Source -> src, Src
- specification, Specification -> spec, Spec
- standard, Standard -> std, Std
- surface, Surface -> surf, Surf
- temporary, Temporary -> temp, Temp
- texture, Texture -> tex, Tex
- transformation, Transformation -> transf, Transf
- vector, Vector -> vec, Vec
- volume, Volume -> vol, Vol

Avoid abbreviations that are ambiguous in context. In particular, do not use
`norm` for `normal`, because it can also mean a vector or matrix norm.

## Variable Suffixes

- `_in`: a function input converted, copied or otherwise prepared internally.
- `_out`: a value constructed for return from a function.
- `_raw`: an unvalidated or unprocessed source value.
- `_std`: a value in Riley's standard convention.
- `_loc`: a local index or value in an element or other containing structure.
- `_glob`: a global index or value in the complete mesh or scene.
