import sys
import numpy as np
import matplotlib.pyplot as plt


data=[]
cpu=[]
gpu=[]
edge=[]
node=[]
speedup=[]

def readData(filename):
	with open(filename) as f:
		line = f.readlines()
		line = [x.strip() for x in line]
		for i in range(len(line)):
			row = line[i].split()
			if i%4==0:
				data.append(row[0])				
			if i%4==1:
				node.append((int)(row[2][1:]))
				edge.append((int)(row[4]))
			if i%4==2:
				cpu.append((float)(row[4]))
			if i%4==3:
				gpu.append((float)(row[5]))
				speedup.append(cpu[i/4]/gpu[i/4]) #old/new
	


def plot():
	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	plt.plot(node, gpu, 'bo')
	for i in range(len(data)):
    		ax.annotate('walk=%s' % data[i], xy=(node[i], gpu[i]), textcoords='data')

	plt.xlabel('number of nodes')
	plt.ylabel('GPU runtime(ms)')
	plt.title('GPU RW Walk=80')
	fig.savefig("gpugraph.jpg")	
	plt.show()

	fig = plt.figure(2)
        ax = fig.add_subplot(111)
        plt.plot(node, speedup, 'ro')
        for i in range(len(data)):
                ax.annotate('%s' % data[i], xy=(node[i], speedup[i]), textcoords='data')

        plt.xlabel('number of nodes')
        plt.ylabel('Speed up ratio')
        plt.title('GPU RW speed up')
        fig.savefig("speedup.jpg")
        plt.show()

def main():
	readData(sys.argv[1])
	plot()


if __name__ == "__main__":
	main()
