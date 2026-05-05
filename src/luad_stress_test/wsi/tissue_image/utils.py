import re


def parser_template(file_name: str, template_string: str) -> dict:
    regex = re.sub(r"{(\w+)}", r"(?P<\1>\\w+)", template_string)
    match = re.match(regex, file_name)

    if match is None:
        raise ValueError("Could not match template to the file name!")

    return match.groupdict()
