import numpy as np
import matplotlib.pyplot as plt


def phase_plot(parameters: dict):
    r1, r2 = parameters["r1"], parameters["r2"]
    K1, K2 = parameters["K1"], parameters["K2"]
    alpha1,alpha2 = parameters["alpha1"], parameters["alpha2"]

    t = parameters["t"]

    x1, x2 = np.linspace(0,10,100), np.linspace(0,10,100)

    nullcline_1 = (r2/alpha2) * (1 - (x2/K2))
    nullcline_2 = (r1/alpha1) * (1 - (x1/K1))

    X1, X2 = np.meshgrid(x1, x2)
    dx1 = r1* X1* (1 - X1/K1) - alpha1* X1* X2
    dx2 = r2* X2* (1 - X2/K2) - alpha2* X2* X1

    plt.figure()

    plt.streamplot(x1,
                   x2,
                   dx1.T,
                   dx2.T,
                   1.2,
                   color = "grey")    
    
    plt.plot(x1,
             nullcline_1,
             "b")
    
    plt.plot(nullcline_2,
             x2,
             "r")

    plt.xlim([0, 10])
    plt.ylim([0, 10])

    plt.title("Phase portrait")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.savefig("./images/null_clines.png")
    

def ode_solver(parameters: dict,
               var: int = 1):
    t = parameters["t"]

    it = len(parameters["x1_0"])

    x1 = np.zeros((it, len(t)))
    x2 = np.zeros((it, len(t)))

    if var == 1:
        for i in range(len(parameters["x1_0"])):
            x1[i, 0] = parameters["x1_0"][i]
            x2[0, 0] = parameters["x2_0"][i]

            for j in range(1,len(t)):
                x1[i, j], x2[i, j] = update_rule(x1[i, j-1],
                                                    x2[i, j-1],
                                                    parameters)
    else:
        for i in range(len(parameters["x1_0"])):
            x1[0, 0] = parameters["x1_0"][i]
            x2[i, 0] = parameters["x2_0"][i]

        for j in range(1,len(t)):
                x1[i, j], x2[i, j] = update_rule(x1[i, j-1],
                                                    x2[i, j-1],
                                                    parameters)

    return x1,x2 


def update_rule(x1: float,
                   x2: float,
                   parameters: dict):   
    r1, r2 = parameters["r1"], parameters["r2"]
    K1, K2 = parameters["K1"], parameters["K2"]
    alpha1, alpha2 = parameters["alpha1"], parameters["alpha2"]
    
    dt = parameters["dt"]

    if parameters["model"] == 1:
        x_1_new = x1 + (r1*x1*(1-(x1/K1)))*dt
        x_2_new = x2 + (r2*x2*(1-(x2/K2)))*dt

    else:
        x_1_new = x1 + (r1*x1*(1-(x1/K1)) - alpha1*x1*x2)*dt
        x_2_new = x2 + (r2*x2*(1-(x2/K2)) - alpha2*x1*x2)*dt
    
    return x_1_new, x_2_new


def plot(x1: np.ndarray,
         x2: np.ndarray,
         parameters: dict,
         var: int = 1):
    t = parameters["t"]

    plt.figure()
    plt.title("Population evolution over time")
    plt.xlabel("t")
    plt.ylabel("Population")

    colors = ["b", "r--", "g-.", "k"]

    if var == 1:
        plt.plot(t,
                 np.repeat(parameters["K1"], len(t)),
                 "k--", label = "Capacity")
        
        for i in range(len(x1[:,0])):
            plt.plot(t,
                     x1[i, :],
                     colors[i],
                     label = f"$x_0 = {x1[i ,0]}$")

    else:
        plt.plot(t,
                 np.repeat(parameters["K2"],len(t)),
                 "k--")
        for i in range(len(x1[:,0])):  
            plt.plot(t,
                     x2[i, :],
                     colors[i],
                     label = f"$x_0 = {x2[i ,0]}$")

    plt.legend()
    plt.savefig(f"images/pop_growth{var}.png")


def main():
    model = 1 #1 Logistic, 2 Lotka--Volterra
    r1,r2 = 1,1.2
    K1,K2 = 5,4.5
    alpha1,alpha2 = 0.5,0.4
    
    x1_0,x2_0 = [0.1, 3, 5, 7], [0.1, 2, 4.5, 6]

    T = 20

    t = np.linspace(0,int(T),100)
    dt = t[1]-t[0]

    parameters = {"r1": r1, "r2": r2,
                "K1": K1, "K2": K2,
                "alpha1": alpha1, "alpha2": alpha2,
                "x1_0":x1_0, "x2_0": x2_0,
                "T": T, "t": t, "dt": dt, "model": model}
    
    for i in range(2):
        x1,x2 = ode_solver(parameters, i+1)

        plot(x1, x2, parameters, i+1)


if __name__ == '__main__':
    main()
