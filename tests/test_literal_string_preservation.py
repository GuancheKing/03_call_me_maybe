from src.generator import preserve_literal_string_value


assert preserve_literal_string_value(
    "Shrek",
    "Greet shrek"
) == "shrek"

assert preserve_literal_string_value(
    "JOHN",
    "Greet john"
) == "john"

assert preserve_literal_string_value(
    "cat",
    "concatenate these strings"
) == "cat"

assert preserve_literal_string_value(
    "hello",
    "Reverse the string 'hello'"
) == "hello"

assert preserve_literal_string_value(
    "data.json",
    "Read DATA.JSON from the folder"
) == "DATA.JSON"

assert preserve_literal_string_value(
    "cat",
    "cat and another cat"
) == "cat"

assert preserve_literal_string_value(
    "",
    "Greet shrek"
) == ""

print("Literal string preservation tests passed.")
