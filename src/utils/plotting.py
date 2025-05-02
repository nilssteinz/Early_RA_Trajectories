from matplotlib import pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans


class plot_trajectory:
    def __init__(self) -> None:
        self.options_edge = {
            "arrowsize": 40,
            "node_size": 800,
            "connectionstyle": "arc3, rad=0.15",
        }

    def _edge_in_digraph(dataset, cutoff=0.15):
        G = nx.DiGraph()
        labels_set = [0 for i in range(len(dataset) ** 2)]
        # print(labels_set)
        for nr, i in enumerate(dataset.round(2).T.iterrows()):
            labels_set[nr] = f"{i[0][1]}"
            for j in i[1].items():
                if i[0][1] == j[0][1]:
                    continue

                if j[1] <= 0:
                    if -j[1] <= cutoff:
                        continue
                else:
                    if j[1] <= cutoff:
                        continue
                # print(j[1])
                G.add_edge((i[0][1]), (j[0][1]), weight=j[1])
        return G, labels_set

    def _define_scaling(scaling):
        kmeans = KMeans(3).fit(scaling.values.reshape(-1, 1))
        # print(kmeans.labels_)
        volgorede_dict = dict(zip(scaling.index, kmeans.labels_))
        # print(volgorede_dict)
        lijst, lijst_fixed = {}, {}
        for i in volgorede_dict:
            lijst[volgorede_dict[i]] = lijst.get(volgorede_dict[i], []) + [i]
        for nr, i in enumerate(lijst):
            lijst_fixed[nr] = lijst[i]
        return lijst_fixed

    def plot(self, probs, scale, title="None", min_prob=0.1):
        cmap = mpl.colormaps["Set1"]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        print(probs)
        B, labels = self._edge_in_digraph(probs, min_prob)
        # print(B.edges(data=True))
        scaling = self._define_scaling(scale)
        left_nodes = [x for x in scaling[0]]
        middle_nodes = [x for x in scaling[1]]
        right_nodes = [x for x in scaling[2]]
        scale_nr = [i[x] for i in scaling.values() for x in range(len(i))]

        pos = {
            n: (scale.loc[(n)], i + (1 / len(left_nodes)))
            for i, n in enumerate(left_nodes)
        }
        pos.update(
            {
                n: (scale.loc[(n)], i + (1 / len(middle_nodes)))
                for i, n in enumerate(middle_nodes)
            }
        )
        pos.update(
            {
                n: (scale.loc[(n)], i + (1 / len(right_nodes)))
                for i, n in enumerate(right_nodes)
            }
        )

        nx.draw_networkx_nodes(
            B,
            pos,
            nodelist=scale.index,
            node_color=cmap(scale.index),
            edgecolors="black",
        )
        m_weights = [W["weight"] for u, v, W in B.edges(data=True)]

        nx.draw_networkx_edges(B, pos, alpha=0.6, width=m_weights, **self.options_edge)
        nx.draw_networkx_labels(B, pos, font_size=22)
        y = pd.DataFrame(pos).mean(axis=1)[0]
        x = pd.DataFrame(pos).mean(axis=1)[0] - 0.1
        plt.arrow(
            x=x, dx=0.2, y=y - 0.8, dy=0, width=0.05, color="black", head_length=0.1
        )
        plt.text(x + 0.1, y - 1.1, "time", style="italic", fontsize=15, color="Black")
        plt.tight_layout()
        plt.axis("off")
        ax.set_title(title)
        return ax


class plot_heatmap:
    pass
