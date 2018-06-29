import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#declare variables
Learning_rate = 0.3 #weird one
Itterations = 200
Max_rand = 200
Min_rand = -200
Plot_times = 15

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "darkgrey",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

def closeto0(): 
    
    # Make 100 phony data points in NumPy.
    data = np.float32(np.random.rand(2, 200)) # Random input

    # Construct a linear model.
    bais = tf.Variable(tf.zeros([1]))
    Weights = tf.Variable(tf.random_uniform([1, 2], Min_rand, Max_rand))
    y = tf.matmul(Weights, data) + bais

    # Minimize the squared errors.
    loss = tf.reduce_mean(tf.square(y))
    optimizer = tf.train.GradientDescentOptimizer(Learning_rate)
    train = optimizer.minimize(loss)

    x = []
    y = []

    # For initializing the variables.
    init = tf.initialize_all_variables()

    # Launch the graph
    sess = tf.Session()
    sess.run(init)

    # Fit the plane.
    for step in range(0, Itterations+1):
        sess.run(train)
        if step % 20 == 0:
            x.append(step)
            y.append(sess.run(bais))

    plt.plot(x, y)

    plt.title("Try to go closer to 0")
    plt.xlabel("itterations")
    plt.ylabel("value")

    plt.xlim(0, Itterations) 
    plt.ylim(Min_rand, Max_rand)

for a in range(0, Plot_times):
    closeto0()

plt.grid()

plt.show()
