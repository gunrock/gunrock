#!/usr/bin/python2

#convert matrix exchange pattern graphs to snap edge list
#for use with PowerGraph

import sys
import random

def parseHeader( line ):
  if not line.startswith( '%%MatrixMarket' ):
    raise ValueError( 'invalid header line: %s' % line )
  matrix, format, edgeData, symType = line[15:].split()
  return { 'format': format, 'edgeData': edgeData, 'symType': symType }


def convert( header, inputFile, outputFile, outputExt ):
  if outputExt == 'mtx':
    commentDelimiter = '%'
    outputFile.write('%%MatrixMarket matrix ' + header['format'] + ' Integer ' + header['symType'] +'\n')
  else:
    commentDelimiter = '#'

  symType = header[ 'symType' ]
  firstDataLine = True
  while 1:
    line = inputFile.readline()
    if not line:
      return
    if line.startswith('%'):
      outputFile.write( commentDelimiter + line[1:] )
    else:
      if firstDataLine: #ignore the first non-comment line
        if outputExt == 'mtx':
          outputFile.write(line) #bogus values that are ignored by our reader

        firstDataLine = False
      else:
        src, dst = line.split()[:2] #any edge values are thrown
        #adjust for matrix format's 1-based indexing
        src = int(src)
        dst = int(dst)
        if outputExt != 'mtx':
          src = src - 1
          dst = dst - 1

        edge_value = random.randint(0, 100)
        if outputExt == 'mtx':
          outputFile.write( '%d %d %d\n' % (src, dst, edge_value) )
        else:
          outputFile.write( '%d %d\n' % (src, dst) )

        if symType == 'symmetric' and outputExt != 'mtx':
            outputFile.write( '%d %d\n' % (dst, src) )
        elif (symType == 'skew-symmetric' or symType == 'hermitian') and outputExt != 'mtx':
            outputFile.write( '%d %d\n' % (dst, src) )


if __name__ == '__main__':
  if len( sys.argv ) != 3:
    print( 'Usage: matrix2snap.py input.mtx output.[mtx, edges]' )
    sys.exit(1)

  inputName, outputName = sys.argv[1:]

  outputExt = outputName.split('.')[-1]

  if outputExt != 'mtx' and outputExt != 'edges':
    print( 'Unknown extension: ', outputExt )
    sys.exit(1)

  input = open( inputName )
  header = parseHeader( input.readline() )
  output = open( outputName, 'w' )
  convert( header, input, output, outputExt )
