"""
Builds task2a.py for you: takes the OFFICIAL task2a.py and replaces ONLY the
block between the two banner lines
    ADD YOUR IMPLEMENTATION HERE   ...   END OF YOUR IMPLEMENTATION
with the code in task2a_functions.py. Everything else in the file is kept
exactly as e-Yantra wrote it.

Run from the task2a folder:
    python merge_2a.py
Your current task2a.py is saved first as task2a_before_merge.py.
"""
import re
import shutil
import sys
from pathlib import Path

OFFICIAL = Path("task2a.py")
MINE = Path("task2a_functions.py")


def main():
    if not OFFICIAL.is_file() or not MINE.is_file():
        print("Run this inside the task2a folder; it needs task2a.py and "
              "task2a_functions.py next to it.")
        return 1

    text = OFFICIAL.read_text(encoding="utf-8")
    # the banner is three lines: ####, #### ADD ... ####, ####
    m_start = re.search(
        r"^#{10,}\n#+ *ADD YOUR IMPLEMENTATION HERE *#+\n#{10,}\n", text, re.M)
    m_end = re.search(
        r"^#{10,}\n#+ *END OF YOUR IMPLEMENTATION *#+\n#{10,}\n", text, re.M)
    if not m_start or not m_end or m_end.start() < m_start.end():
        print("Could not find the two banner lines in task2a.py. Restore the "
              "official file first:  git checkout -- task2a.py")
        return 1

    shutil.copyfile(OFFICIAL, "task2a_before_merge.py")
    body = MINE.read_text(encoding="utf-8").strip("\n")
    merged = (text[:m_start.end()] + "\n" + body + "\n\n\n"
              + text[m_end.start():])
    OFFICIAL.write_text(merged, encoding="utf-8")
    compile(merged, "task2a.py", "exec")          # syntax check
    print("Done. task2a.py now holds your implementation "
          "({} lines).".format(merged.count("\n")))
    print("The previous file was saved as task2a_before_merge.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
