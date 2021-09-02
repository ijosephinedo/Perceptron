from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

# Root frame in GUI
root = Tk()
root.title("Perceptron")
root.geometry('800x450')
#root.resizable(False, False)
f_inputData = Frame(root)
f_graphs = Frame(root)
f_inputData.pack(side=RIGHT)
f_graphs.pack(side=LEFT, fill=BOTH, expand=1)

# Perceptron variables
classX = IntVar()
classX.set(0)
eta = DoubleVar()
eta.set(0.1)
maxEpochs = StringVar()
maxEpochs.set("30")
trainingSet = []
desiredSet = []
weights = []
done = False

# Perceptron functions
def init_Weights():
	for i in range(3):
		weights.append(np.random.uniform(-1,1))
	x = np.arange(-5, 5, 0.01)
	ax_p.plot(x, (-weights[1] * x + weights[0])/weights[2], color='#feffb3')
	fig_p.canvas.draw()
	print(weights)
	b_weights['state'] = 'disable'
	rb_class1['state'] = 'disable'
	rb_class2['state'] = 'disable'

def perceptron():
	global l_epochNumber
	global done
	done = False
	nEpoch = 0
	x = np.arange(-5, 5, 0.01)
	while not done and (nEpoch < int(maxEpochs.get()) ):
		done = True
		activation_function = 0
		for i in range(len(trainingSet)):
			for j in range(len(weights)):
				activation_function += trainingSet[i][j] * weights[j]
			if activation_function >= 0:
				activation_function = 1
			else:
				activation_function = 0
			error = desiredSet[i] - activation_function
			if error != 0:
				for j in range(len(weights)):
					weights[j] = weights[j] + eta.get() * error * trainingSet[i][j]
				done = False
				print(weights)
				input("Continuar")
		if not done:
			ax_p.plot(x, (-weights[1] * x + weights[0])/weights[2], color='#81b1d2')
		else:
			ax_p.plot(x, (-weights[1] * x + weights[0])/weights[2], color='#b3de69')
		#toDelete = hyperplane.pop(0)
		#toDelete.remove()
		fig_p.canvas.draw()
		input("Siguiente epoca")
		nEpoch+=1
		l_epochNumber.destroy()
		l_epochNumber = Label(master=lf_perceptron, text="# Epoch: " + str(nEpoch))
		l_epochNumber.pack(anchor=W, padx=5, pady=5)

def start_over():
	global done
	ax_p.clear()
	start_graph()
	fig_p.canvas.draw()
	trainingSet.clear()
	weights.clear()
	done = False
	b_weights['state'] = 'normal'
	rb_class1['state'] = 'normal'
	rb_class2['state'] = 'normal'

def start_graph():
	plt.axhline(color='white', lw=1)
	plt.axvline(color='white', lw=1)
	plt.title("Perceptron hyperplane")
	plt.xlim(-5, 5)
	plt.ylim(-5, 5)
	plt.grid(True, linestyle='--', color='gray')

def onclick(event):
	#['#8dd3c7', '#feffb3', '#bfbbd9', '#fa8174', '#81b1d2', '#fdb462', '#b3de69', '#bc82bd', '#ccebc4', '#ffed6f'])
	global done
	goodPoint = True
	if event.xdata is None or event.ydata is None:
		goodPoint = False
	if classX.get() == 0 and goodPoint:
		plt.plot(event.xdata, event.ydata, color='#fa8174', marker='o')
		trainingSet.append([-1, event.xdata, event.ydata])
		desiredSet.append(0)
		fig_p.canvas.draw()
	if classX.get() == 1 and goodPoint:
		plt.plot(event.xdata, event.ydata, color='#8dd3c7', marker='o')
		trainingSet.append([-1, event.xdata, event.ydata])
		desiredSet.append(1)
		fig_p.canvas.draw()
	if done:
		y = (-weights[1] * event.xdata + weights[0])/weights[2]
		if y > event.ydata:
			colorX = '#8dd3c7'
		else:
			colorX = '#fa8174'
		plt.plot(event.xdata, event.ydata, color=colorX, marker='o')
		fig_p.canvas.draw()


# GUI - Initial data
lf_weights = LabelFrame(f_inputData, text="Initial data", padx=5, pady=5)
b_weights = Button(master=lf_weights, text="Random weights", command=init_Weights)
rb_class1 = Radiobutton(master=lf_weights, text="Class 1", variable=classX, value=0)
rb_class2 = Radiobutton(master=lf_weights, text="Class 2", variable=classX, value=1)

# GUI - Hyperparameters
lf_hyperParam = LabelFrame(f_inputData, text="Hyperparameters", padx=5, pady=5)
l_maxEpochs = Label(master=lf_hyperParam,  text="Max Epochs: ")
e_maxEpochs = Entry(master=lf_hyperParam, textvariable=maxEpochs)
l_learningRate = Label(master=lf_hyperParam,  text="Learning Rate: ")
s_learningRate = Scale(master=lf_hyperParam, variable=eta, resolution=0.1, from_=0.1, to=0.9, orient=HORIZONTAL)

# GUI - Perceptron
lf_perceptron = LabelFrame(f_inputData, text="Perceptron", padx=5, pady=5)
b_perceptron = Button(master=lf_perceptron, text="Start Perceptron", command=perceptron)
b_restart = Button(master=lf_perceptron, text="Restart", command=start_over)
l_epochNumber = Label(master=lf_perceptron, text="# Epoch: ")

# GUI - Canvas
plt.style.use('dark_background')
fig_p, ax_p = plt.subplots(figsize=(8,8))
start_graph()
cid = fig_p.canvas.mpl_connect('button_press_event', onclick)
canvas_p = FigureCanvasTkAgg(fig_p, master=f_graphs)  # A tk.DrawingArea.
canvas_p.get_tk_widget().pack(padx=5, pady=5)

# Packing - Initial data
lf_weights.pack(fill=BOTH, padx=5, pady=5)
b_weights.pack(side=BOTTOM, fill=X, padx=5, pady=5)
rb_class1.pack(side=LEFT, padx=5, pady=5)
rb_class2.pack(padx=5, pady=5)

# Packing - Hyperparameters
lf_hyperParam.pack(fill=BOTH, padx=5, pady=5)
l_learningRate.pack(anchor=SW)
s_learningRate.pack(fill=X, anchor=NW, padx=5, pady=5)
l_maxEpochs.pack(anchor=SW)
e_maxEpochs.pack(anchor=NW, padx=5, pady=5)

# Packing - Perceptron
lf_perceptron.pack(fill=BOTH, padx=5, pady=5)
b_perceptron.pack(fill=BOTH, padx=5, pady=5)
b_restart.pack(side=BOTTOM, fill=BOTH, padx=5, pady=5)
l_epochNumber.pack(anchor=W, padx=5, pady=5)

root.mainloop()
