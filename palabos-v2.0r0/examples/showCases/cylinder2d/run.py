import subprocess
import sys

# Parameters
N = sys.argv[1]
Re = sys.argv[2]
lx = sys.argv[3]
ly = sys.argv[4]

args = ("./cylinder2d", N, Re, lx, ly)
popen = subprocess.Popen(args, stdout=subprocess.PIPE)
popen.wait()
print("Done generating data")
