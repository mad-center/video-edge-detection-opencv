import os
from pathlib import Path

mypath = "/tmp/d/a.dat"
filename = Path(mypath).name
head, tail = os.path.split(mypath)
filename_noext = Path(mypath).stem

print(filename)
print(head, tail)
print(filename_noext)
