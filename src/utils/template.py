import re

template_regex = re.compile(r'\{\{(\w[\w\d]+)\}\}')


def parse_template(text, data):
    def repl(m):
        if isinstance(data, dict):
            return data[m.group(1)]
        else:
            return getattr(data, m.group(1))

    return template_regex.sub(repl, text)
