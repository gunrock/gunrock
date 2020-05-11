#!/usr/bin/python2

# convert snap to mtx

import sys
import random

def convert(inputFile, outputFile, outputExt):
  commentDelimiter = '%'
  outputFile.write('%%MatrixMarket matrix coordinate pattern symmetric'+'\n')

  firstDataLine = True
  while 1:
    line = inputFile.readline()
    if not line:
      return
    if line.startswith('%'):
      outputFile.write(commentDelimiter + line[1:])
    elif line.startswith('#'):
      outputFile.write(commentDelimiter + line[1:])
    else:
      src, dst = line.split()[:2] #any edge values are thrown
      #adjust for matrix format's 1-based indexing
      src = int(src)
      dst = int(dst)

      src = src + 1
      dst = dst + 1

      if src != dst:
        outputFile.write('%d %d\n' % (dst, src))

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print('Usage: snap2matrix.py input.edges output.mtx')
    sys.exit(1)

  inputName, outputName = sys.argv[1:]

  outputExt = outputName.split('.')[-1]

  if outputExt != 'mtx':
    print('Unknown extension: ', outputExt)
    sys.exit(1)

  input = open(inputName)
  output = open(outputName, 'w')
  convert(input, output, outputExt)
