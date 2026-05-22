import re

with open('notebooks/01_acquisition.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

# Pattern for git conflict markers on execution_count
pattern = re.compile(r'<<<<<<< HEAD\n\s*"execution_count": \d+,\n=======\n\s*"execution_count": \d+,\n>>>>>>> [^\n]+\n')
text = pattern.sub('"execution_count": 1,\n', text)

with open('notebooks/01_acquisition.ipynb', 'w', encoding='utf-8') as f:
    f.write(text)

print("Conflict fixed.")
