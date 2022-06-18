import os.path

import pandas as pd
import plotly.express as px

import constants as ct
from formatter import DatasetFormatter

frmt = DatasetFormatter()


def get_plot_path(plot_name: str) -> str:
    return os.path.join(ct.VIS_PLOTS_DIR, f"{plot_name}.html")


def visualize_dataset(df: pd.DataFrame):
    px.histogram(df, x=ct.PASSENGER_ID, facet_row=ct.SURVIVED).write_html(get_plot_path('scatter_id'))

    px.parallel_categories(frmt.cabine_to_class(df), color=ct.SURVIVED)\
        .write_html(get_plot_path('parallel_coordinates_cabines'))

    px.parallel_categories(df,
                           dimensions=[ct.PCLASS, ct.SEX, ct.EMBARKED, ct.PARCH, ct.SIBSP, ct.SURVIVED],
                           color=ct.AGE,
                           color_continuous_scale=px.colors.sequential.Inferno)\
        .write_html(get_plot_path('parallel_categories_age'))

    px.parallel_categories(df,
                           dimensions=[ct.PCLASS, ct.SEX, ct.EMBARKED, ct.PARCH, ct.SIBSP, ct.SURVIVED],
                           color=ct.FARE,
                           color_continuous_scale=px.colors.sequential.Inferno) \
        .write_html(get_plot_path('parallel_categories_fare'))

    px.parallel_categories(df,
                           dimensions=[ct.PCLASS, ct.SEX, ct.EMBARKED],
                           color=ct.SURVIVED,
                           color_continuous_scale=px.colors.sequential.Inferno) \
        .write_html(get_plot_path('parallel_categories_survived'))

    px.parallel_coordinates(df,
                            color=ct.PARCH)\
        .write_html(get_plot_path('parallel_coordinates_survived'))

    px.scatter(df, x=ct.FARE, y=ct.AGE, color=ct.SURVIVED, size=ct.PCLASS, log_x=True) \
        .write_html(get_plot_path('scatter_fare_age_pclass'))

    px.histogram(df, x=ct.AGE, color=ct.SURVIVED, facet_row=ct.PCLASS) \
        .write_html(get_plot_path('hist_age'))

    px.histogram(df, x=ct.FARE, color=ct.SURVIVED, facet_row=ct.PCLASS) \
        .write_html(get_plot_path('hist_fare'))

    scatter_df = df.replace(ct.MALE, 1).replace(ct.FEMALE, 2)
    scatter_df[ct.SURVIVED] = scatter_df[ct.SURVIVED].astype(str)
    px.scatter(scatter_df,
               x=ct.FARE, y=ct.AGE, color=ct.SURVIVED, log_x=True, facet_row=ct.SEX) \
        .write_html(get_plot_path('scatter_fare_age_sex'))
