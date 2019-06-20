import re


def labelize(text, remove_space=True):
    BEGIN, INSIDE, END, SINGLE, EXTRA = "B", "I", "E", "S", "X"

    def _is_space(_current_char):
        return _current_char == " "

    def _is_single_char(_current_char, _previous_char, _next_char):
        return _previous_char == " " and _next_char == " " and _current_char != " "

    def _is_begin_char(_current_char, _previous_char, _next_char):
        return _previous_char == " " and _next_char != " " and _current_char != " "

    def _is_end_char(_current_char, _previous_char, _next_char):
        return _previous_char != " " and _next_char == " " and _current_char != " "

    def _is_inside_char(_current_char, _previous_char, _next_char):
        return _previous_char != " " and _next_char != " " and _current_char != " "

    _text = text

    original_text = re.sub(' +', ' ', _text.strip())
    labelized_text = ""

    for i, t in enumerate(original_text):
        current_char = t
        previous_char = original_text[i - 1] if i > 0 else " "
        next_char = original_text[i + 1] if i < len(original_text) - 1 else " "

        if _is_space(current_char):
            labelized_text += EXTRA
        elif _is_single_char(current_char, previous_char, next_char):
            labelized_text += SINGLE
        elif _is_begin_char(current_char, previous_char, next_char):
            labelized_text += BEGIN
        elif _is_end_char(current_char, previous_char, next_char):
            labelized_text += END
        elif _is_inside_char(current_char, previous_char, next_char):
            labelized_text += INSIDE
        else:
            raise ValueError()

    if remove_space:
        original_text, labelized_text = original_text.replace(' ', ''), labelized_text.replace('X', '')

    return original_text, labelized_text


def remove_multiple_spaces(text):
    return re.sub(' +', ' ', text)
