import re

def clean_line(line):
    return " ".join(line.split(" ")[2:])

def ignore_line(line):
    first_part = line[:2]
    return first_part != "tx" and first_part != "rx"

def next_good_line(f):
    for line in f:
        clean = clean_line(line)
        if ignore_line(clean):
            continue
        return clean
    return None

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