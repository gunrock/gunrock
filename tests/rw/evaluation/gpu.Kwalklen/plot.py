import sys
import numpy as np
import matplotlib.pyplot as plt

runtime=[]
walk=[10, 160, 20, 40, 5, 80]

def readData(filename):
	with open(filename) as f:
		line = f.readlines()
		line = [x.strip() for x in line]
		for i in range(len(line)):
			runtime.append((float)(line[i].split()[5]))


def plot():
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(walk, runtime, 'bo')
	for i in range(len(walk)):
    		ax.annotate('walk=%s' % walk[i], xy=(walk[i], runtime[i]), textcoords='data')

	plt.xlabel('walk length')
	plt.ylabel('GPU runtime(ms)')
	plt.title('GPU RW K walk')
	fig.savefig("k_walk.jpg")	
	plt.show()

def main():
	readData(sys.argv[1])
	plot()


if __name__ == "__main__":
	main()
