# AUTOGENERATED! DO NOT EDIT! File to edit: 01_sensors.ipynb (unless otherwise specified).

__all__ = ['__construct_INFO__', '__construct_list_DATA__', 'read_files', '__dates_sorted__', '__ind_chrono_order__',
           'sort_chrono', '__elements_vides__', 'traitement_elements_vides', '__condition_DATA__',
           'list_names_channels', 'extract_T0', 'construction_list_df_messung', 'concat_df_messung',
           'traitement_colonnes_zeit', '__clean_name__', 'traitement_colonnes_names', 'patch_buggs_in_column_names',
           'conversion_float', 'change_overhead_values', 'downsample', 'get_automatic_data', 'define_index',
           'get_manual_data', 'df_to_m', 'get_data']

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

# Cell
def __construct_INFO__(name_file):
  """
  Lit le fichier 'structure' et retourne la liste INFO
  """

  f_struc = open(name_file, "r")
  reader = csv.reader(f_struc, delimiter = '\t')

  info = []       #Liste des info contenues dans le fichier permettera d'accéder aux données dans le ficher brut
  for row in reader:
    if len(row) >= 2:
      info.append(row[1])
    else:
      info.append("XXX")

  return info

# Cell
def __construct_list_DATA__(name_file):
  """
  Lit un fichier '.asc' de mesures et retourne une liste image du fichier
  """

  f_csv = open(name_file, "r")
  reader = csv.reader(f_csv, delimiter = '\t')

  DATA = []    #Liste du fichier texte
  for row in reader:
    DATA.append(row)

  return DATA

# Cell
def read_files(list_files_names,
               structure_data):
  """
  Retourne le tuple des listes issues de la lecture des fichiers :
  - list_data : liste des listes images des fichiers de mesure 'messung_XX'
  - info : liste des informations contenues sur les premières lignes des fichiers de mesures
  """

  info = []
  list_data = []

  info = __construct_INFO__(structure_data)

  for file_name in list_files_names:
    list_data.append(__construct_list_DATA__(file_name))

  return list_data, info

# Cell
def __dates_sorted__(list_data, info):
  """
  Construit un objet provisoire [(date.datetime, i_file)]
  """
  i_date = info.index('Date')
  dateFormatter = "%m/%d/%Y"

  dates = []

  for i_file in range(len(list_data)):
    file_data = list_data[i_file]
    dates_typ = datetime.strptime(file_data[2][0], dateFormatter)

    dates.append( (dates_typ , i_file) )

  dates_sorted = sorted(dates, key=lambda date: date[0])

  return dates_sorted

# Cell
def __ind_chrono_order__(dates_sorted):
  """
  Renvoie la liste des indices des fichiers dans l'ordre chronologique
  """
  ind_order = [elt[1] for elt in dates_sorted]

  return ind_order

# Cell
def sort_chrono(list_data, info):
  dates_sorted = __dates_sorted__(list_data, info)
  i_new_order = __ind_chrono_order__(dates_sorted)

  new_list = []
  for i in i_new_order:
    new_list.append(list_data[i])

  return new_list

# Cell
# unit function

def __elements_vides__(data):
  """
  Supprime les élèments vides en dernière position
  """

  list_data = copy(data)     # Permet de ne pas modifier DATA

  for i_row in range(len(list_data)):
    row = list_data[i_row]

    if len(row) > 0:
      last_elt = row[-1]

      if last_elt == '':
        list_data[i_row] = row[:-1]

  return list_data

# Cell
# global function

def traitement_elements_vides(data):
  """
  Permet d'appliquer la fonction __elements_vides__ à la liste DATA_MESSUNG
  """
  #Bricolage, mais la structure de DATA_MESSUNG fait que data[0][0] est
  #la liste image du fichier est une chaine de caractère

  if data[0][0] == 'HBM_CATMAN_DATAFILE_40':
    data = __elements_vides__(data)

  else:
    for i in range(len(data)):
      data[i] = __elements_vides__(data[i])

  return data

# Cell
def __condition_DATA__(file_data):
  """
  Teste si DATA est bien une liste image d'un fichier de mesures
  """

  if file_data[0][0] == 'HBM_CATMAN_DATAFILE_40': #Une meilleure condition pourra etre trouvée
    return True

  else:
    return False

# Cell
def list_names_channels(data_list, info):
  """
  Construction de la liste des noms des capteurs
  """
  assert __condition_DATA__(data_list), 'Fonction uniquement applicable à un fichier DATA'

  i_names_channels = info.index('Name Channel')
  names_channels = copy(data_list[i_names_channels])

  return names_channels

# Cell
def extract_T0(file_data, info):
  """
  Renvoie un objet datatime
  """
  str_T0 = file_data[info.index('T0')][0]

  # Extraction des elements
  split_T0 = str_T0.split(' ')
  split_day = split_T0[1]

  day = split_day.split('=')[1]
  time = split_T0[2] + ' ' + split_T0[3]

  day_time = day + ' ' + time
  # Creation de l'objet datetime
  dateFormatter = "%m/%d/%Y %I:%M:%S %p"
  time = datetime.strptime(day_time, dateFormatter)

  return time

# Cell
def construction_list_df_messung(data_messung,
                                 info):
  """
  Construit et renvoie la liste des DataFrame des séries de mesures en parcourant DATA_MESSUNG
  """
  #indices début des mesures
  i_mesures = info.index('YYY')

  # On vérifie que la partie commentaires de DATA_MESSUNG est de la meme taille
  assert len(data_messung[0][:i_mesures+1]) == len(info), "Mauvais fichier DATA_MESSUNG"


  num_tot_files = len(data_messung)
  list_df = []

  for num_file in range(num_tot_files):
    data_file = data_messung[num_file]
    names_channels = list_names_channels(data_file, info)

    #On récupère la partie comportant les mesures
    list_mesures = data_file[i_mesures:]

    #Creation de la base de données du ficher des mesures courantes
    df_num_file = pd.DataFrame(list_mesures, columns=names_channels)

    # # #Colonnes paramètres de l'expérience
    # df_num_file['Serie de mesures'] = num_file

    # Colonne temps global
    time_0 = extract_T0(data_file, info)
    freq = df_num_file['Zeit  1 - Standardmessrate']

    times = [time_0 + timedelta(seconds=float(delta_t)) for delta_t in freq]
    df_num_file['date'] = times

    df_num_file.set_index('date',inplace = True)

    list_df.append(df_num_file)

  return list_df

# Cell
def concat_df_messung(list_df_messung_temp):
    df_concat = list_df_messung_temp[0]
    for df_serie in list_df_messung_temp[1:]:
        df_concat = pd.concat([df_concat, df_serie])

    return df_concat

# Cell
def traitement_colonnes_zeit(df_messung):
  """
  Retire les colonnes inutlies de la df (Zeit)
  """

  for column_name in df_messung.columns:
    if 'Zeit' in column_name:
      del df_messung[column_name]

  return df_messung

# Cell
def __clean_name__(name_column_CH):
  """
  Argument : string de la forme CH_Y_SENSOR_INDICE  CH=Z
  Renvoie la string CH_Y
  """

  assert type(name_column_CH) == str
  assert 'CH' in name_column_CH

  new_name = ''

  splited_name = name_column_CH.split(' ')
  CH_Y_SENSOR_INDICE = splited_name[0] #Drop de CH=Z
  dropped_part = splited_name[1]

  if 'CH' in dropped_part:
    SENSOR_INDICE = CH_Y_SENSOR_INDICE[5:]

    new_name = SENSOR_INDICE

  else :
    return name_column_CH #La string a déja été traitée

  return new_name

# Cell
def traitement_colonnes_names(df_messung):
  """
  Renomme les colonnes pour afficher le nom des capteurs
  """
  new_names = []

  for column_name in df_messung.columns:
    if 'CH' in column_name:
      new_name = __clean_name__(column_name)
      new_names.append(new_name)

    else:
      new_names.append(column_name)

  df_messung.columns = new_names

  return df_messung

# Cell
def patch_buggs_in_column_names(df, date_of_switch_sensor_B):

  # CH_4_EH13 is missing on the second Station
  #TODO

  #'EV31' in t.columns but it should be 'EH31'
  df = df.rename(columns={'EV31': 'EH31'})

  # Sensor B has been switched with sensor B2 from 25.05.2021
  df['EDS_B2'] = df['EDS_B']
  df.loc[df.index < date_of_switch_sensor_B, "EDS_B2"] = np.nan
  df.loc[df.index >= date_of_switch_sensor_B, "EDS_B"] = np.nan


  return df

# Cell
def conversion_float(df):
  """
  Transforme les éléments en float
  """
  for column_name in df.columns:
    df[column_name] = df[column_name].astype(float)

  return df

# Cell
def change_overhead_values(df):
  df = df.replace(-1000000.0, np.nan)
  return df

# Cell
def downsample(df):
  return df.resample('d').mean()

# Cell
def get_automatic_data(list_files_names,
                       structure_data
                       ):
  """
  Fonction merge DATA --> df_finale
  """
  # Creation des listes images des fichiers
  list_files_names, info = read_files(list_files_names, structure_data)

  list_files_names = sort_chrono(list_files_names, info)

  # Suppression des éléments vides
  list_files_names = traitement_elements_vides(list_files_names)

  # Construction de la liste des df
  list_df_messung = construction_list_df_messung(list_files_names, info)

  # Concatenation des df
  df_messung = concat_df_messung(list_df_messung)

  # Mise en forme des colonnes
  df_messung = traitement_colonnes_zeit(df_messung)
  df_messung = traitement_colonnes_names(df_messung)
  df_messung = conversion_float(df_messung)

  # clear bugg
  df_messung = patch_buggs_in_column_names(df_messung, '2021-07-15')

  # clean
  df_messung = conversion_float(df_messung)
  df_messung = change_overhead_values(df_messung)
  df_messung = downsample(df_messung)


  return df_messung

# Cell
def define_index(df):
  df_time = copy(df)

  if 'date' in df.columns:

    df_time['Datetime'] = pd.to_datetime(df_time['date'], format='%d.%m.%Y')
    df_time = df_time.set_index('Datetime')
    df_time = df_time.drop(['date'], 1)

  return df_time

# Cell
def get_manual_data(file_name):

  #Lecture du fichier CSV
  df_hand = pd.read_csv(file_name, delimiter=';')

  #Creation de l'index date
  df_hand = define_index(df_hand)

  # defined for automatic values
  df_hand = conversion_float(df_hand)

  return df_hand

# Cell
def df_to_m(df):
  df_m = copy(df)

  for name_column in df_m.columns:
    if 'Temp' not in name_column:
      df_m[name_column] = df_m[name_column] / 1e6

  return df_m

# Cell
def get_data(list_files_names,
             structure_data,
             file_name_df_hand,
             ):
  """
  Fonction merge DATA --> df_finale
  """
  # Get automatic data
  df_automatic = get_automatic_data(list_files_names, structure_data)

  # Get manual data
  df_hand = get_manual_data(file_name_df_hand)

  # Merge
  ## fill na to avoid getting new columns
  for col in ['W11', 'W21']:
    df_automatic[col].fillna(df_hand[col], inplace=True)
    df_hand.drop(columns=col, inplace=True)

  ## merge
  df_messung = pd.merge(df_hand, df_automatic, left_index=True, right_index=True, how='outer')

  #df_messung = pd.concat([df_automatic, df_hand])

  # Change units
  df_messung = df_to_m(df_messung)

  # Sort values
  df_messung = df_messung.sort_index()

  return df_messung