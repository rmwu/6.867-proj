import matplotlib.pyplot as plt

def plot(x,train,validate,
         xlabel,ylabel,title,
         filename):
    tplt = plt.semilogx(x, train, basex=2, label="training data",
                 marker=".", color="r")
    vplt = plt.semilogx(x, validate, basex=2, label="validation data",
                 marker=".", color="g")
    
    plt.legend(loc="center right")
    
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    
    plt.savefig(filename, format='pdf')
    plt.show()