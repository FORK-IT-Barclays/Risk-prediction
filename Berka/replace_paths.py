import pathlib, sys

# Base absolute paths to replace (both with and without the 'VECTOR' segment)
BASE_PATHS = [
    r'.',
    r'.',
]
REPLACEMENT = '.'

root = pathlib.Path(r'.')

for p in root.rglob('*'):
    if p.is_file():
        try:
            # Attempt to read as text (ignore binary files)
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue
        new_text = text
        for base in BASE_PATHS:
            if base in new_text:
                new_text = new_text.replace(base, REPLACEMENT)
        if new_text != text:
            try:
                p.write_text(new_text, encoding='utf-8')
            except Exception:
                pass
print('All path replacements completed.')
