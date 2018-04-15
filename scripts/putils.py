"""
Coding: UTF-8
Indentation : 4spaces
"""
import matplotlib.pyplot as plt


def to_header(msg, dec='#'):
    """ Given a string, it returns a header-like string"""

    m = len(msg) 
    m = m // len(dec) -1

    return f"{dec*m+dec*7}\n{dec*2} {msg} {dec*2}\n{dec*m+dec*7}"


def plot_graphs(graphs, titles, FILENAME="./", save=False, show=True):
    """ Plot n graphs with the specified titles
    :param graphs: list(np.array) graphs to be plot
    :param titles: list(string) titles of the graphs
    :return:
    """
    plt.subplots(figsize=(15,5))
    for i, (graph, title) in enumerate(zip(graphs, titles)):

        plt.subplot(1, len(graphs), i+1)
        plt.imshow(graph)
        plt.title(title)

    #plt.suptitle(titles[-1])
    plt.tight_layout()
    if save:
        plt.savefig(FILENAME)#, dpi=400)

    if show:
        plt.show()
    plt.close()

