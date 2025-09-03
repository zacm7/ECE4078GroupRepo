"""
Safe bulk renamer for all pibot_dataset folders.
Renames image files to img_162.ext, img_163.ext, ... per folder.
Performs two-stage renaming to avoid collisions.

Usage: python scripts/rename_pibot_images.py
"""
from pathlib import Path
import re
import os

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
START_INDEX = 162

nsort_re = re.compile(r'(\d+)|\D+')

def natural_key(s):
    parts = nsort_re.findall(s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def rename_in_dir(d: Path):
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        return None
    files.sort(key=lambda p: natural_key(p.name))

    # Build final names
    mapping = {}
    idx = START_INDEX
    for p in files:
        mapping[p] = d / f"img_{idx}{p.suffix.lower()}"
        idx += 1

    # If any target name equals a source name but mapped differently, we'll use two-stage rename.
    # First stage: rename all source files to a temp name.
    temp_names = {}
    for i, src in enumerate(mapping.keys()):
        temp = d / f".__tmp_rename_{i}__{src.suffix.lower()}"
        temp_names[src] = temp

    # Perform temp renames
    for src, tmp in temp_names.items():
        try:
            src.rename(tmp)
        except Exception as e:
            return (False, f"Failed to rename {src} -> {tmp}: {e}")

    # Perform final renames
    results = []
    for src, final in mapping.items():
        tmp = temp_names[src]
        try:
            # If final exists and is not one of our tmp files, fail to avoid clobbering.
            if final.exists() and final not in temp_names.values():
                return (False, f"Target exists and would be clobbered: {final}")
            tmp.rename(final)
            results.append((tmp.name, final.name))
        except Exception as e:
            return (False, f"Failed to rename {tmp} -> {final}: {e}")

    return (True, results)


def main():
    root = ROOT
    found = list(root.rglob('pibot_dataset'))
    if not found:
        print('No pibot_dataset folders found under', root)
        return

    summary = {}
    for d in found:
        res = rename_in_dir(d)
        summary[str(d)] = res

    # Print concise report
    for d, res in summary.items():
        print('Folder:', d)
        if res is None:
            print('  No image files found.')
            continue

        # Handle error tuple (False, message)
        if isinstance(res, tuple):
            ok, data = res
            if ok is False:
                print('  ERROR:', data)
                continue
            results = data
        else:
            results = res

        # results should be a list of (oldname, newname)
        print(f'  Renamed {len(results)} files:')
        for old, new in results[:10]:
            print('   ', old, '->', new)
        if len(results) > 10:
            print('   ...', len(results) - 10, 'more')
    print('\nDone.')

if __name__ == '__main__':
    main()
