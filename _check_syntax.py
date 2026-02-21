import ast

files = [
    'face_engine/__init__.py',
    'face_engine/detector.py',
    'face_engine/lighting.py',
    'face_engine/quality.py',
    'face_engine/skin_tone.py',
    'face_engine/tryon.py',
    'face_engine/hud.py',
    'main.py',
    'tools/csv_filter.py',
]

all_ok = True
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as fh:
            src = fh.read()
        ast.parse(src)
        print(f'  OK  {f}')
    except SyntaxError as e:
        print(f'  FAIL {f}: {e}')
        all_ok = False

print()
print('Result: ALL SYNTAX VALID' if all_ok else 'Result: ERRORS FOUND')
