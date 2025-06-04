import inspect


def evaluate_variable(variable):

    print()
    frame = inspect.stack()[1]
    filename = frame.filename
    line_number = frame.lineno

    print(f'Evaluating variable from "{filename}", line {line_number}.')
    print(f"Type  : {type(variable)}")

    as_string = repr(variable)

    if len(as_string) > 100:
        as_string = as_string[:100] + " - [CAPPED AT 100 CHARACTERS]"

    print(f"Value : {as_string}")

    try:
        print(f"Len   : {len(variable)}")
    except:
        print(f"Len   : NA")

    try:
        print(f"Dtype : {variable.dtype}")
    except:
        print(f"Dtype : NA")

    try:
        print(f"Shape : {variable.shape}")
    except:
        print(f"Shape : NA")

    try:
        print(f"Size  : {variable.size}")
    except:
        print(f"Size  : NA")
