#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

from phenograph.cluster import cluster as cluster2
from sknetwork.clustering import Louvain, get_modularity


class trajectory_seq_analysis:
    def __init__(
        self,
        data: pd.DataFrame,
        label_col: str = "label",
        time_col: str = "days",
        pat_id: str = "PATNR",
    ) -> None:
        self.data = data.copy()
        self.label_col = label_col
        self.time_col = time_col
        self.pat_id = pat_id
        self.volgorde = pd.Index(
            sorted(self.data[label_col].unique(), reverse=True)
        ).astype(int)
        self.data["dest"] = (
            self.data.sort_values(time_col)
            .groupby(pat_id)[[label_col]]
            .shift(-1, axis=0, fill_value=-1)
        )
        self.count_matrix = pd.DataFrame()

    def _count_matrix(
        self, df: pd.DataFrame = None, query: str = "ilevel_0 in ilevel_0"
    ) -> pd.DataFrame:
        if not df.items():
            df = self.data
        count_dict = pd.DataFrame.from_dict(
            {x: {x: 0 for x in self.volgorde} for x in self.volgorde}
        )
        for nr, point in self.data.query(query).iterrows():
            from_, to_ = point[[self.label_col, "dest"]]
            if to_ == -1:
                continue
            count_dict[from_][to_] += 1
        self.count_matrix = count_dict
        return count_dict

    def _calc_costs(self, counts: pd.DataFrame = None):
        if type(counts) == None:
            self._count_matrix()
            counts = self.count_matrix

        counts = counts.sum(axis=0) / counts.sum().sum()
        counts = -np.log(counts)
        scaler = MinMaxScaler(feature_range=(1, 10))
        scaler = scaler.fit_transform(counts.values.reshape(-1, 1))
        counts = pd.DataFrame(scaler, index=self.volgorde)
        counts.index = counts.index.astype(str)
        counts
        sub = counts * -0.5
        insert = sub * 0.75
        
        
        return counts, sub, insert

    def __calc_value(
        self,
        part_seq_1: list,
        part_seq_2: list,
    ):
        if part_seq_1 == part_seq_2:
            return self.counts.loc[part_seq_1, 0]
        elif part_seq_1 != part_seq_2:
            return self.sub.loc[part_seq_1, 0]

    def matrix(
        self,
        seq1: str,
        seq2: str,
    ) -> np.ndarray:
        m: np.ndarray = np.zeros([len(seq1) + 1, len(seq2) + 1])
        m[:, 0] = np.cumsum([0] + [self.insert.loc[x, 0] for x in seq1])
        m[0, :] = np.cumsum([0] + [self.insert.loc[x, 0] for x in seq2])
        for nr_i in range(len(seq1) + 1):
            if nr_i == 0:
                continue
            for nr_j in range(len(seq2) + 1):
                if nr_j == 0:
                    continue
                diag = m[nr_i - 1, nr_j - 1] + self.__calc_value(
                    seq1[nr_i - 1], seq2[nr_j - 1]
                )
                down = m[nr_i - 1, nr_j] + self.insert.loc[seq2[nr_j - 1], 0]
                up = m[nr_i, nr_j - 1] + self.insert.loc[seq1[nr_i - 1], 0]
                # print(nr_i,nr_j, seq1[nr_i-1],seq2[nr_j-1])
                # print(diag,down, up)
                m[nr_i, nr_j] = max(diag, down, up)
        return m

    def norm_sim(self, seq1, seq2, mode: str = "global") -> float:
        seq1, seq2 = seq1[0], seq2[0]
        seq1 = list(map(str, seq1))
        seq2 = list(map(str, seq2))
        # print(seq1, seq2)
        if mode == "global":
            score = self.matrix(seq1, seq2)[-1, -1]
        elif mode == "local":
            score = self.matrix(seq1, seq2).max()
        # print(score)
        scale_max = (sum([self.counts.loc[x, 0] for x in seq1]) / 2) + (
            sum([self.counts.loc[x, 0] for x in seq2]) / 2
        )
        scale_min = (
            sum([self.sub.loc[x, 0] for x in seq1[: min(len(seq1), len(seq2))]]) / 2
            + sum([self.sub.loc[x, 0] for x in seq2[: min(len(seq1), len(seq2))]])
            / 2
            + sum(
                [self.sub.loc[x, 0] for x in seq1[min(len(seq1), len(seq2)) :]]
                + [self.sub.loc[x, 0] for x in seq2[min(len(seq1), len(seq2)) :]]
            )
        )
        score = max(score,scale_min)
        return (score - scale_min) / (scale_max - scale_min)

    def get_labels(self, array: np.ndarray, algo: str = "leiden", k=150):
        if algo == "louvain":
            louvain_algo = Louvain(random_state=41)
            lv = louvain_algo.fit(array)
            labels = lv.labels_
        elif algo == "leiden":
            labels, graph, modularity = cluster2(
                array,
                k=k,
                seed=20221003,
                clustering_algo="leiden",
            )
        labels = pd.DataFrame(labels, columns= ["traject_labels"]) + 1
        return labels

    def get_trajectories(
        self,
        data: pd.DataFrame = None,
        query: str = "ilevel_0 in ilevel_0",
        algo="leiden",
        k=150,
    ):
        if type(data) != pd.DataFrame:
            data = self.data
        data_query = data.query(query).copy()

        traject_counts = self._count_matrix(data_query, query=query)
        traject_prob = (traject_counts / traject_counts.sum(axis=0)).round(2)
        traject_test = traject_counts * traject_prob
        scaling = (
            self.data.groupby(self.label_col)[self.time_col].mean().loc[self.volgorde]
        )
        scaling = (scaling - scaling.min()) / (scaling.max() - scaling.min())

        global counts, sub, insert
        self.counts, self.sub, self.insert = self._calc_costs(traject_counts)
        seq_list = data_query.pivot_table(
            values=[self.label_col], index=["PATNR"], aggfunc=list
        ).reset_index()
        sim_array = squareform(
            pdist(
                seq_list[[self.label_col]],
                self.norm_sim,
            )
        )

        seq_list["traject_label"] = self.get_labels(sim_array, algo=algo, k=k)
        self.seq_list = seq_list
        self.sim_array = sim_array
        return seq_list, sim_array


if __name__ == "__main__":
    print("Main")
    datasamples_use = pd.read_csv(
        "/exports/reum/nsteinz/results/" + "/predictie/predictie_data.csv",
    )
    # print(datasamples_use)
    sample = datasamples_use.groupby("PATNR").head(1).head(400).PATNR
    datasamples_use = datasamples_use.query("PATNR in @sample")
    # print(sample)
    trajectory = trajectory_seq_analysis(
        datasamples_use,
    )
    print(trajectory.get_trajectories())
