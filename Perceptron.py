from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

RED = "#fa8174"
YELLOW = "#feffb3"
GREEN = "#b3de69"
BLUE = "#81b1d2"

# Root frame in GUI
root = Tk()
root.title("Perceptron")
root.geometry('600x600')
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
training = []
desired = []
weights = []
done = False
error_array = []


# Perceptron functions
def init_Weights():
    for i in range(3):
        weights.append(np.random.uniform(-1, 1))
    hyperplane(YELLOW)
    print("Initial weigths: ")
    print(weights)
    b_weights['state'] = 'disable'
    rb_class1['state'] = 'disable'
    rb_class2['state'] = 'disable'
    b_perceptron['state'] = 'normal'


def hyperplane(color_line, style_line='-'):
    x = np.arange(-5, 5, 0.01)
    y = (-weights[1] * x + weights[0]) / weights[2]
    ax_p[0].plot(x, y, color=color_line, linestyle=style_line)
    fig_p.canvas.draw()


def activation_function(v):
    if v >= 0:
        return 1
    else:
        return 0


def perceptron():
    global done
    errors = 0
    done = False
    epochs = 0
    print("Training Set...")
    print(training)
    print("Desired set...")
    print(desired)
    while not done and (epochs < int(maxEpochs.get())):
        done = True
        epochs += 1
        errors = 0
        print("Época: " + str(epochs))
        for i in range(len(training)):
            weighted_sum = 0
            for j in range(len(weights)):
                weighted_sum += training[i][j] * weights[j]
            error = desired[i] - activation_function(weighted_sum)
            print(str(i) + ',' + str(error))
            if error != 0:
                for j in range(len(weights)):
                    weights[j] = weights[j] + eta.get() * error * training[i][j]
                done = False
                errors += 1
                print(weights)
        error_array.append(errors)
        if not done:
            hyperplane(BLUE, ':')
        else:
            hyperplane(BLUE, '-')
    x = np.arange(1, epochs + 1)
    x_ticks = np.arange(1, epochs + 1, 1)
    ax_p[1].set_xticks(x_ticks)
    ax_p[1].bar(x, error_array)
    fig_p.canvas.draw()
    b_perceptron['state'] = 'disable'


def start_over():
    global done
    print("Restarting figure...")
    ax_p[0].clear()
    ax_p[1].clear()
    start_graph()
    fig_p.canvas.draw()
    training.clear()
    print("Restarting Training Set...")
    print(training)
    desired.clear()
    print("Restarting Desired Set...")
    print(desired)
    weights.clear()
    print("Restarting weights...")
    print(weights)
    error_array.clear()
    print("Restarting errors...")
    print(error_array)
    done = False
    b_weights['state'] = 'normal'
    rb_class1['state'] = 'normal'
    rb_class2['state'] = 'normal'


def start_graph():
    major_ticks = np.arange(-5, 6, 1)
    ax_p[0].set_xticks(major_ticks)
    ax_p[0].set_yticks(major_ticks)
    ax_p[0].grid(which='major', linestyle=':', alpha=0.5)
    ax_p[0].axhline(color='white', lw=1)
    ax_p[0].axvline(color='white', lw=1)
    ax_p[0].set_title("Perceptron hyperplane")
    ax_p[1].set_title("Errors per epoch")
    ax_p[0].set_xlim(-5, 5)
    ax_p[0].set_ylim(-5, 5)


def onclick(event):
    #['#8dd3c7', '#feffb3', '#bfbbd9', '#fa8174', '#81b1d2', '#fdb462',
    #'#b3de69', '#bc82bd', '#ccebc4', '#ffed6f'])
    global done
    goodPoint = True
    if event.xdata is None or event.ydata is None:
        goodPoint = False
    if classX.get() == 0 and goodPoint:
        ax_p[0].plot(event.xdata, event.ydata, color=RED, marker='o')
        training.append([-1, event.xdata, event.ydata])
        desired.append(0)
        fig_p.canvas.draw()
    if classX.get() == 1 and goodPoint:
        ax_p[0].plot(event.xdata, event.ydata, color=GREEN, marker='o')
        training.append([-1, event.xdata, event.ydata])
        desired.append(1)
        fig_p.canvas.draw()
    if done:
        y = (-weights[1] * event.xdata + weights[0]) / weights[2]
        if desired[0] == 0:
            if y > event.ydata:
                colorX = GREEN
            else:
                colorX = RED
        else:
            if y > event.ydata:
                colorX = RED
            else:
                colorX = GREEN
        ax_p[0].plot(event.xdata, event.ydata, color=colorX, marker='o')
        fig_p.canvas.draw()


# GUI - Initial data
lf_ws = LabelFrame(f_inputData, text="Initial data", padx=5, pady=5)
b_weights = Button(master=lf_ws, text="Random weights", command=init_Weights)
rb_class1 = Radiobutton(master=lf_ws, text="Class 0", variable=classX, value=0)
rb_class2 = Radiobutton(master=lf_ws, text="Class 1", variable=classX, value=1)

# GUI - Hyperparameters
lf_hyperP = LabelFrame(f_inputData, text="Hyperparameters", padx=5, pady=5)
l_maxEpochs = Label(master=lf_hyperP, text="Max Epochs: ")
e_maxEpochs = Entry(master=lf_hyperP, textvariable=maxEpochs, width=15)
l_learningRate = Label(master=lf_hyperP, text="Learning Rate: ")
s_learningRate = Scale(master=lf_hyperP,
                       variable=eta,
                       resolution=0.1,
                       from_=0.1,
                       to=0.9,
                       orient=HORIZONTAL)

# GUI - Perceptron
lf_perceptron = LabelFrame(f_inputData, text="Perceptron", padx=5, pady=5)
b_perceptron = Button(master=lf_perceptron,
                      text="Start Perceptron",
                      command=perceptron,
                      state='disable')
b_restart = Button(master=lf_perceptron, text="Restart", command=start_over)
l_epochNumber = Label(master=lf_perceptron, text="# Epoch: ")

# GUI - Canvas
plt.style.use('dark_background')
fig_p, ax_p = plt.subplots(2,
                           figsize=(8, 12),
                           gridspec_kw={'height_ratios': [4, 1]},
                           constrained_layout=True)
start_graph()
cid = fig_p.canvas.mpl_connect('button_press_event', onclick)
canvas_p = FigureCanvasTkAgg(fig_p, master=f_graphs)  # A tk.DrawingArea.
canvas_p.get_tk_widget().pack(padx=5, pady=5)

# Packing - Initial data
lf_ws.pack(fill=BOTH, padx=5, pady=5)
b_weights.pack(side=BOTTOM, fill=X, padx=5, pady=5)
rb_class1.pack(anchor=W, padx=5, pady=5)
rb_class2.pack(anchor=W, padx=5, pady=5)

# Packing - Hyperparameters
lf_hyperP.pack(fill=BOTH, padx=5, pady=5)
l_learningRate.pack(anchor=SW)
s_learningRate.pack(fill=X, padx=5, pady=5)
l_maxEpochs.pack(anchor=SW)
e_maxEpochs.pack(anchor=NW, padx=5, pady=5)

# Packing - Perceptron
lf_perceptron.pack(fill=BOTH, padx=5, pady=5)
b_perceptron.pack(fill=BOTH, padx=5, pady=5)
l_epochNumber.pack(anchor=W, padx=5, pady=5)
b_restart.pack(side=BOTTOM, fill=BOTH, padx=5, pady=5)

root.mainloop()
