import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors, KDTree, kneighbors_graph
from sklearn.metrics import pairwise_distances

from lifelines.utils import datetimes_to_durations
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts


from lifelines import CoxPHFitter


class surv_functions:
    ALL_QUERY = "PATNR in PATNR"

    def km_data(
        dataset: pd.DataFrame,
        group_name: str,
        datum_column: str,
        query_sort: str,
        query_differ: str,
    ) -> np.array:
        """
        This function creates the correct datarepresentation for the module 'lifelines'
        it uses query language from pandas to select the correct reprensentation of the different groups.


        """
        start = (
            dataset.sort_values(datum_column)
            .query(query_sort)
            .groupby(group_name)[[group_name, datum_column]]
            .head(1)
            .copy()
            .rename(columns={datum_column: "start"})
        )
        last = (
            dataset.sort_values(datum_column)
            .query(query_sort)
            .groupby(group_name)[[group_name, datum_column]]
            .tail(1)
            .copy()
            .rename(columns={datum_column: "last_visit"})
        )
        end = (
            dataset.sort_values(datum_column)
            .query(query_sort + " & " + query_differ)
            .groupby(group_name)[[group_name, datum_column]]
            .head(1)
            .copy()
            .rename(columns={datum_column: "end"})
        )
        start = start.merge(last, on=group_name, how="left")
        start["last"] = (start.last_visit - start.start).dt.days
        start = start.merge(end, on=group_name, how="left")
        T, E = datetimes_to_durations(start.start, start.end, freq="D")
        start["Time"] = T
        start["Event"] = E
        start["Time"] = start[["last", "Time"]].min(axis=1).astype(int)
        return start


print("imports worked")
