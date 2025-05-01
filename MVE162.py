import numpy as np
import matplotlib.pyplot as plt


def update_rule(x1: float,
                x2: float,
                parameters: dict):
    model = parameters["model"]

    r1, r2 = parameters["r1"], parameters["r2"]
    K1, K2 = parameters["K1"], parameters["K2"]
    alpha1, alpha2 = parameters["alpha1"], parameters["alpha2"]
    
    dt = parameters["dt"]

    if model == 1:
        return x1 + r1* x1* (1 - x1/K1)* dt,\
            x2 + r2* x2* (1 - x2/K2)* dt
    else:
        return x1 + r1* x1* (1 - x1/K1)* dt - alpha1* x1* x2* dt,\
            x2 + r2* x2* (1 - x2/K2)* dt - alpha2* x1* x2* dt


def solver(parameters: dict):
    t = parameters["t"]

    it = len(parameters["x1_0"])

    x1 = np.zeros((it, len(t)))
    x2 = np.zeros((it, len(t)))

    for i in range(it):
        x1[i, 0] = parameters["x1_0"][i]
        x2[i, 0] = parameters["x2_0"][i]

        for j in range(1,len(t)):
            x1[i, j], x2[i, j] = update_rule(x1[i, j-1],
                                             x2[i, j-1],
                                             parameters)

    return x1,x2 


def plot(x1: np.ndarray,
         x2: np.ndarray,
         parameters: dict):
    
    t = parameters["t"]
    model = parameters["model"]

    if model == 1:
        plt.figure()
        plt.title("Population evolution over time")
        plt.xlabel("t")
        plt.ylabel("Population")

        colors = ["b", "r--", "g-.", "k"]

        plt.plot(t, np.repeat(parameters["K1"], len(t)), "k--", label = "Capacity")
        
        for i in range(len(x1[:,0])):
            plt.plot(t, x1[i, :], colors[i], label = f"$x_0 = {x1[i ,0]}$")

        plt.legend()
        plt.savefig(f"images/logistic_model_1.png")


        plt.figure()
        plt.title("Population evolution over time")
        plt.xlabel("t")
        plt.ylabel("Population")

        colors = ["b", "r--", "g-.", "k"]

        plt.plot(t, np.repeat(parameters["K2"], len(t)), "k--", label = "Capacity")
        
        for i in range(len(x2[:,0])):
            plt.plot(t, x2[i, :], colors[i], label = f"$x_0 = {x2[i ,0]}$")

        plt.legend()
        plt.savefig(f"images/logistic_model_2.png")
    
    else:
        pass
        """
        plt.figure()

        plt.streamplot(x1, x2, dx1.T, dx2.T, 1.2, color = "grey")    
        
        plt.plot(x1, nullcline_1, "b")
        
        plt.plot(nullcline_2, x2, "r")

        plt.xlim([0, 10])
        plt.ylim([0, 10])

        plt.title("Phase portrait")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

        plt.savefig("./images/null_clines.png")
        """


def main():
    model = 1 #1 Logistic, 2 Lotka--Volterra

    r1,r2 = 1,1.2
    K1,K2 = 5,4.5
    alpha1,alpha2 = 0.5,0.4
    
    if model == 1:
        x1_0, x2_0 = [0.1, 3, 5, 7], [0.1, 2, 4.5, 6]
    else:
        x1_0, x2_0 = 3, 2

    T = 20

    t = np.linspace(0,int(T),100)
    dt = t[1]-t[0]

    parameters = {"r1": r1, "r2": r2,
                "K1": K1, "K2": K2,
                "alpha1": alpha1, "alpha2": alpha2,
                "x1_0":x1_0, "x2_0": x2_0,
                "T": T, "t": t, "dt": dt, "model": model}
    
    x1, x2 = solver(parameters)

    plot(x1, x2, parameters)

    

if __name__ == "__main__":
    main()