from src.decoder import (
    is_a_valid_integer,
    is_a_valid_number,
    is_a_valid_numeric_value,
)


assert is_a_valid_integer("0")
assert is_a_valid_integer("23")
assert is_a_valid_integer("-23")

assert not is_a_valid_integer("23.5")
assert not is_a_valid_integer("-23.5")
assert not is_a_valid_integer("")
assert not is_a_valid_integer("-")
assert not is_a_valid_integer("01")


assert is_a_valid_number("23")
assert is_a_valid_number("23.5")
assert is_a_valid_number("-23.5")


assert is_a_valid_numeric_value("23", "integer")
assert not is_a_valid_numeric_value("23.5", "integer")

assert is_a_valid_numeric_value("23", "number")
assert is_a_valid_numeric_value("23.5", "number")

assert not is_a_valid_numeric_value("23", "string")


print("Integer and number validation tests passed.")
