import re

# Cleans up junk from lines
def clean_line(line):
    return " ".join(line.split(" ")[2:])

# Ignores lines that do not begin with rx or tx
def ignore_line(line):
    first_part = line[:2]
    return first_part != "tx" and first_part != "rx"

# Tests a log file with a regex pattern, returns true if it passes
def check_log(name, pattern):
    with open(name) as f:
        cleaned = ""
        for line in f:
            clean = clean_line(line)
            if ignore_line(clean):
                continue
            cleaned += clean
            
        match = re.match(pattern, cleaned)
        if match is None:
            print("Test failed.")
            return False
        else:
            print("Test passed.")
            return True