import re
import shutil
from pathlib import Path

FILE = Path("manga_translator/translators/chatgpt.py")

print("Opening:", FILE)

if not FILE.exists():
    print("ERROR: chatgpt.py not found")
    exit()

# backup
backup = FILE.with_suffix(".py.bak")
shutil.copy(FILE, backup)
print("Backup created:", backup)

code = FILE.read_text(encoding="utf-8")

changes = 0

def replace(pattern, repl):
    global changes
    new, n = re.subn(pattern, repl, code, flags=re.MULTILINE)
    if n:
        print("Replaced", n, "occurrences of:", pattern)
    return new

# --- 1 ChatCompletions -> Responses API ---
new_code = re.sub(
    r"self\.client\.chat\.completions\.create",
    "self.client.responses.create",
    code
)
if new_code != code:
    print("Converted chat.completions.create -> responses.create")
    changes += 1
code = new_code

# --- 2 messages= -> input= ---
new_code = re.sub(
    r"messages=",
    "input=",
    code
)
if new_code != code:
    print("Converted messages= -> input=")
    changes += 1
code = new_code

# --- 3 max_completion_tokens -> max_output_tokens ---
new_code = re.sub(
    r"max_completion_tokens",
    "max_output_tokens",
    code
)
if new_code != code:
    print("Converted max_completion_tokens -> max_output_tokens")
    changes += 1
code = new_code

# --- 4 response.choices[0].message.content -> response.output_text ---
new_code = re.sub(
    r"response\.choices\[0\]\.message\.content",
    "response.output_text",
    code
)
if new_code != code:
    print("Converted response parsing")
    changes += 1
code = new_code

# --- 5 choices check ---
new_code = re.sub(
    r"if not response\.choices:",
    "if not response.output_text:",
    code
)
if new_code != code:
    print("Converted response empty check")
    changes += 1
code = new_code

FILE.write_text(code, encoding="utf-8")

print()
print("DONE")
print("Total change blocks:", changes)
print("Backup saved as:", backup)