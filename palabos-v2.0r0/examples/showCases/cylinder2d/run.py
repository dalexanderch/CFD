import subprocess
args = ("./cylinder2d", "100", "600", "2", "1")
#Or just:
#args = "bin/bar -c somefile.xml -d text.txt -r aString -f anotherString".split()
popen = subprocess.Popen(args, stdout=subprocess.PIPE)
popen.wait()
output = popen.stdout.read()
print (output)
