# remove_even_lines_keyboard.py
"""
Removes every even-numbered line (2, 4, 6, ...) from keyboard.csv.
Usage: python remove_even_lines_keyboard.py input.csv output.csv
"""
import sys

if len(sys.argv) != 3:
    print("Usage: python remove_even_lines_keyboard.py input.csv output.csv")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for idx, line in enumerate(fin, 1):
        if idx % 2 != 0:
            fout.write(line)
print(f"Done. Output written to {output_file}")
