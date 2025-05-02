import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import fisher_exact, chi2
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
from IPython.display import display, Markdown, display_markdown
import seaborn as sns


class Propensity:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        time_column: str,
        label_column: str,
        patient_indetifier: str,
        med_array: list[str] = [],
        query_name: str = "",
    ) -> None:
        self.additive = query_name
        self.indentifier = patient_indetifier
        self.time = time_column
        self.label_col = label_column
        self.dataframe = dataframe.copy()
        self.end_symbol = {int(len(self.dataframe[label_column].unique())): "E"}
        self.labels = sorted(self.dataframe[label_column].unique(), reverse=True)
        self.dataframe["dest"] = (
            self.dataframe.sort_values(self.time)
            .groupby(self.indentifier)[label_column]
            .shift(-1, axis=0, fill_value=-1)
        )
        self.med_count = {}
        self.med_prob = {}
        self.propensity = {}
        self.p_values = {}
        self.significant = {}

        self.get_infos()

    def get_infos(self):
        self.count_patients = len(self.dataframe[self.indentifier].unique())

    def _fisher_trajectory(self, med: str):
        """

        [  x (a)          n - x    (b) ]
        [N - x (c)    M - (n + N) + x  (d)]
        """
        dataframe_array = self.med_count[med]
        fishers_array = [[0 for x in self.labels] for x in self.labels]
        for i in self.labels:
            for j in self.labels:
                try:
                    x = dataframe_array.values[i - 1, j - 1]
                    N = dataframe_array.values[:, j - 1].sum()
                    n = self.count_dict.values[i - 1, j - 1]
                    M = self.count_dict.values[:, j - 1].sum()
                    fisher_table = np.array(
                        [
                            x,  # x (a)
                            n - x,  #  n - x (b)
                            N - x,  # N - x (c)
                            M - (N + n) + x,  # M  # N + n  # M - (n + N) + x  (d)
                        ]
                    ).reshape(2, 2)
                    # print( i,j, "\n",fisher_table)
                    fishers_array[i - 1][j - 1] = fisher_exact(fisher_table)[1]
                except IndexError as e:
                    print(e)
                    print(i, j)
                    print(dataframe_array)
                    fishers_array[i - 1][j - 1] = np.NaN

        return pd.DataFrame(fishers_array, columns=self.labels, index=self.labels)

    def _calc_transition_matrix(self, query: str = "days>=0"):
        count_dict = pd.DataFrame.from_dict(
            {x: {x: 0 for x in self.labels} for x in self.labels}
        )
        for nr, point in self.dataframe.query(query).iterrows():
            from_, to_ = point[[self.label_col, "dest"]]
            if to_ == -1:
                continue
            count_dict.loc[to_, from_] += 1
        return count_dict

    def calc_transition_matrix(self, query: str = "days>=0"):
        self.count_dict = self._calc_transition_matrix(query=query)
        self.traject_props = self.count_dict / self.count_dict.sum(axis=0)

    def calc_propencetie(self, med: str):
        self.med_count[med] = self._calc_transition_matrix(query=f"`{med}` == 1")
        self.med_prob[med] = self.med_count[med] / self.med_count[med].sum(axis=0)
        self.propensity[med] = self.med_prob[med] / self.traject_props
        pass

    def calc_significants(self, med: str):
        self.p_values[med] = self._fisher_trajectory(med)
        self.significant[med] = pd.DataFrame(
            fdrcorrection(self.p_values[med].values.reshape(1, -1)[0])[0].reshape(
                len(self.labels), len(self.labels)
            ),
            columns=self.labels,
            index=self.labels,
        )

    def visualize_prop(self, med: str) -> mpl.pyplot:
        fig, axs = plt.subplots(1, 1, figsize=(12, 10))
        prop = self.propensity[med]
        # print(prop)
        ax = sns.heatmap(
            pd.DataFrame(
                np.log2(prop).round(2), columns=prop.columns, index=prop.index
            ),
            vmin=-1,
            vmax=1,
            cmap="vlag",
            annot=False,
            ax=axs,
        )
        ax.set(
            xlabel="from",
            ylabel="to",
            title=f"{self.additive} {med.lower()} log$_{2}$(propensity) (n={self.med_count[med].sum().sum()}/{self.count_dict.sum().sum()})",
        )
        for i in self.labels:  # from
            for j in self.labels:  # to
                count = self.count_dict.loc[j, i]
                med_count = self.med_count[med].loc[j, i]
                if prop.loc[j, i] == 0:
                    prop_text = -np.inf
                else:
                    prop_text = np.log2(prop.loc[j, i]).round(2)
                text = f"{med_count}/{count}\n{prop_text}"
                i_, j_ = len(self.labels) - (i), len(self.labels) - (j)
                ax.annotate(
                    text,
                    (i_ + 0.5, j_ + 0.5),
                    color="black",
                    weight="bold",
                    fontsize=10,
                    ha="center",
                    va="center",
                )
                if self.significant[med].loc[j, i] == True:  # corrected pos
                    ax.add_patch(
                        Rectangle(
                            (i_ + 0.025, j_ + 0.025),
                            0.94,
                            0.94,
                            ec="black",
                            fc="None",
                            lw=5,
                            hatch="",
                            zorder=3,
                        )
                    )
                    continue
                elif self.p_values[med].loc[j, i] < 0.05:  # uncorrected pos
                    ax.add_patch(
                        Rectangle(
                            (i_ + 0.025, j_ + 0.025),
                            0.94,
                            0.94,
                            ec="grey",
                            fc="None",
                            lw=5,
                            hatch="",
                            zorder=2,
                        )
                    )

    def describe_data_latex(self, features, query: str = None, index=True):
        if query == None:
            query == f"{self.time} >= 0"
        dataset = self.dataframe.query(query)
        _ = dataset.groupby(self.indentifier).count()

        print(f"{'features'*index} & {self.additive} ")
        print(f"N= & {len(self.dataframe)} ")
        print(
            f"visit counts & {_.median()} ({round(_.quantile(q=0.25),1)} - { round(_.quantile(q=0.75),1)}) "
        )

        for i in features:
            # print(dataset[i])
            if dataset[i].dtypes == "object":
                continue
            elif dataset[i].dtypes == "bool":
                print(
                    f"{i*index} {'(%)'*index} & {round(dataset[i].sum(),1)} ({round(dataset[i].mean()*100, 1)}) "
                )
            else:
                conf = stats.norm.interval(
                    0.95, loc=dataset[i].mean(), scale=dataset[i].std(ddof=1)
                )
                print(
                    f"{i*index} {'(±IQR)'*index} & {round(dataset[i].median(),1)}  ({round(dataset[i].quantile(q=0.25),1)} - { round(dataset[i].quantile(q=0.75),1)}) "
                )

    def print_describe_markdown(self, features, query: str = None, index=True):
        if query == None:
            query == f"{self.time} >= 0"
        dataset = self.dataframe.query(query)

        main_text = ""
        main_text += f"{'|features'*index} | {self.additive} | \n"
        main_text += "|:-------|-------:|\n"
        main_text += f"| N= | {len(dataset)} |\n"
        for i in features:
            # print(dataset[i])
            if dataset[i].dtypes == "object":
                continue
            elif dataset[i].dtypes == "bool":
                main_text += f"|{i*index} {'(%)'*index} | {round(dataset[i].sum(),1)} ({round(dataset[i].mean()*100, 1)})| \n"
                pass
            else:
                main_text += f"|{i*index} {'(±IQR)'*index} | {round(dataset[i].median(),1)}  ({round(dataset[i].quantile(q=0.25),1)} - { round(dataset[i].quantile(q=0.75),1)})| \n"
                pass
        return display_markdown(main_text, raw=True)
