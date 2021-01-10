import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def ChartDrawer(xAxis, YAxis):
    plt.figure(figsize=(15, 10), dpi=100, linewidth=2)
    plt.plot(xAxis, YAxis[0], 's-', color='r', label="tree")
    plt.plot(xAxis, YAxis[1], 's-', color='g', label="planet")
    plt.plot(xAxis, YAxis[2], 's-', color='b', label="beach")
    plt.title("Purity chart", x=0.5, y=1.03)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("count", fontsize=30, labelpad=15)
    plt.ylabel("purity", fontsize=30, labelpad=20)
    plt.legend(loc="best", fontsize=20)
    plt.show()
