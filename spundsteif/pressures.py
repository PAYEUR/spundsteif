# AUTOGENERATED! DO NOT EDIT! File to edit: 03_pressures.ipynb (unless otherwise specified).

__all__ = ['construct_df_csv', 'clean', 'merge_piezo_excavation', 'clean_manual_df', 'get_manual_data',
           'merge_manual_automatic']

# Cell
import pandas as pd
import numpy
import io
import csv
from datetime import datetime

from google.colab import files

from copy import copy

from datetime import timedelta
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go

import numpy as np

from spundsteif import sensors

# Cell
def construct_df_csv(file_name):
  df = pd.read_csv(file_name, delimiter=';')

  #remove empty lines and columns
  df = df.dropna(how='all')
  df = df.dropna(how='all', axis='columns')

  return df

# Cell
def clean(df):

  # dtypes
  df['date'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')

  # remove 'Zeit' and 'Datum'
  df = df.drop(columns=['Datum', 'Zeit'])

  df = df.set_index('date')

  return df

# Cell
def merge_piezo_excavation(df_piezo, df_excavation):
  df = df_piezo.join(df_excavation, how='outer')
  return df

# Cell
def clean_manual_df(df):
  df['GWT [m]'] = df['Tiefe/Top Spundwand [m]']
  df = df.drop(columns=['Tiefe/Top Spundwand [m]'])
  return df

# Cell
def get_manual_data(piezo_file_name, excavation_file_name):
  df_piezo = construct_df_csv(piezo_file_name)
  df_piezo = clean(df_piezo)

  df_excavation = construct_df_csv(excavation_file_name)
  df_excavation = clean(df_excavation)

  df = merge_piezo_excavation(df_piezo, df_excavation)
  df = clean_manual_df(df)
  df = upsample_manual_df(df)
  df = interpolate_manual_df(df)

  return df

# Cell
def merge_manual_automatic(df_manual, df_automatic):
    df = df_automatic.join(df_manual, how='outer')
    return df