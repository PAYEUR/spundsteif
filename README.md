# Spundsteif



> Analyse the experimental data of the project Spundsteif.

`spundsteif` is a library that allows to clean and analyse experimental data coming from the instrumented site during the project Spundsteif.

## Features


`spundsteif` offers the following possibilities:  

+ **Analyse** of mechanical parameters computed from experimental data, with the help of a **dashboard**

+ **Provides a solid documented, unit-tested library** that can be reused for the cleaning of various raw data of measurements

## Installing

> git clone [https://github.com/PAYEUR/spundsteif](https://github.com/PAYEUR/spundsteif)   > pip install -e spundsteif

*Note that `spundsteif` must be installed into the same python environment that you use for both your Jupyter Server and your workspace.*

## How to use

### Dashboard usage

Open the notbook `core.ipyb` and follow the instructions to analyse your data and get the values of the mechanical parameters.

### Library usage

For example let us get a cleaned `pandas.dataframe` from a bunch of raw data.
> one file in *.txt* format describes the position of each sensor on a *x,y,z* grid  
> one file in *.csv* format describes manual measurements performed on sensors  
> some files in *.ASC* format describe automatic measurements performed on sensors



Import library and desired function

```
from spundsteif.sensors import run_computation
```

Upload raw data as files  
in this example we will use test data

```
# Geometry file
STRUCTURE_FILE = './test/1_structure_file.txt'

# Automatic measurements files
AUTOMATIC_FILES= ['./test/1_data_test_01.ASC', './test/1_data_test_02.txt']

# Hand measured data
HAND_FILE = './test/df_hand_test.csv'
```

Get structured and cleaned `pandas.dataframe`

```
df = run_computation(list_files_names=AUTOMATIC_FILES,
                     structure_data=STRUCTURE_FILE, 
                     file_name_df_hand=HAND_FILE)
```

```
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EDS_A</th>
      <th>EDS_B</th>
      <th>EDS_C</th>
      <th>EDS_D</th>
      <th>EDS_E</th>
      <th>EDS_F</th>
      <th>EDS_G</th>
      <th>Temp_a</th>
      <th>Temp_b</th>
      <th>EH11</th>
      <th>EV12</th>
      <th>EH21</th>
      <th>EV22</th>
      <th>EV23</th>
      <th>EV31</th>
      <th>EV32</th>
      <th>EV33</th>
      <th>VH11</th>
      <th>W12</th>
      <th>VH21</th>
      <th>W22</th>
      <th>VH31</th>
      <th>W32</th>
      <th>EH12</th>
      <th>EH13</th>
      <th>EH22</th>
      <th>EH23</th>
      <th>EH32</th>
      <th>EH33</th>
      <th>EV11</th>
      <th>EV21</th>
      <th>VH12</th>
      <th>VH13</th>
      <th>VH22</th>
      <th>VH23</th>
      <th>VH32</th>
      <th>VH33</th>
      <th>W11</th>
      <th>W13</th>
      <th>W21</th>
      <th>W23</th>
      <th>W31</th>
      <th>W33</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-07-07 13:24:40.200000</th>
      <td>-0.007821</td>
      <td>2.159839</td>
      <td>-0.036016</td>
      <td>-1.0</td>
      <td>0.012440</td>
      <td>-6.198071</td>
      <td>0.020038</td>
      <td>12.34039</td>
      <td>18.53559</td>
      <td>0.004134</td>
      <td>0.003226</td>
      <td>0.005147</td>
      <td>0.004189</td>
      <td>0.004093</td>
      <td>0.004815</td>
      <td>0.005774</td>
      <td>0.005584</td>
      <td>0.042620</td>
      <td>-0.000722</td>
      <td>0.042329</td>
      <td>0.004415</td>
      <td>0.005715</td>
      <td>-0.023867</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-07-07 18:24:44.200860</th>
      <td>0.000183</td>
      <td>1.934437</td>
      <td>-0.020863</td>
      <td>-1.0</td>
      <td>0.019554</td>
      <td>-6.212432</td>
      <td>0.030182</td>
      <td>12.36769</td>
      <td>18.45851</td>
      <td>0.004182</td>
      <td>0.003267</td>
      <td>0.005185</td>
      <td>0.004286</td>
      <td>0.004040</td>
      <td>0.004915</td>
      <td>0.005817</td>
      <td>0.005630</td>
      <td>0.042619</td>
      <td>-0.000651</td>
      <td>0.042328</td>
      <td>0.004461</td>
      <td>0.005758</td>
      <td>-0.021864</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-07-07 23:24:49.801710</th>
      <td>-0.008862</td>
      <td>1.705553</td>
      <td>-0.026352</td>
      <td>-1.0</td>
      <td>0.010957</td>
      <td>-6.214493</td>
      <td>0.017809</td>
      <td>12.39935</td>
      <td>18.38660</td>
      <td>0.004136</td>
      <td>0.003212</td>
      <td>0.005134</td>
      <td>0.004263</td>
      <td>0.004086</td>
      <td>0.004900</td>
      <td>0.005769</td>
      <td>0.005586</td>
      <td>0.042619</td>
      <td>-0.000663</td>
      <td>0.042328</td>
      <td>0.004404</td>
      <td>0.005699</td>
      <td>-0.020916</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-07-08 04:24:53.602570</th>
      <td>-0.011061</td>
      <td>1.723809</td>
      <td>-0.037398</td>
      <td>-1.0</td>
      <td>0.008058</td>
      <td>-6.196866</td>
      <td>0.010308</td>
      <td>12.30714</td>
      <td>18.27804</td>
      <td>0.004134</td>
      <td>0.003200</td>
      <td>0.005123</td>
      <td>0.004283</td>
      <td>0.003924</td>
      <td>0.004914</td>
      <td>0.005763</td>
      <td>0.005582</td>
      <td>0.042621</td>
      <td>-0.000563</td>
      <td>0.042330</td>
      <td>0.004391</td>
      <td>0.005689</td>
      <td>-0.020005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2021-07-08 09:24:58.203420</th>
      <td>-0.007851</td>
      <td>1.962178</td>
      <td>-0.032659</td>
      <td>-1.0</td>
      <td>0.011167</td>
      <td>-6.175929</td>
      <td>-0.058488</td>
      <td>12.36118</td>
      <td>18.17512</td>
      <td>0.004160</td>
      <td>0.003219</td>
      <td>0.005140</td>
      <td>0.004328</td>
      <td>0.003960</td>
      <td>0.004958</td>
      <td>0.005784</td>
      <td>0.005419</td>
      <td>0.042620</td>
      <td>-0.000595</td>
      <td>0.042329</td>
      <td>0.004408</td>
      <td>0.005704</td>
      <td>-0.019212</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2021-07-21 00:00:00.000000</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.005671</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.006620</td>
      <td>-0.00509</td>
      <td>NaN</td>
      <td>-0.005513</td>
      <td>-0.004670</td>
      <td>NaN</td>
      <td>-0.004198</td>
      <td>-0.002872</td>
      <td>-0.004110</td>
      <td>NaN</td>
      <td>-0.003165</td>
      <td>-0.005094</td>
      <td>-0.003556</td>
      <td>-0.004029</td>
      <td>-0.003539</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.004647</td>
    </tr>
    <tr>
      <th>2021-07-28 00:00:00.000000</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.005546</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.006407</td>
      <td>NaN</td>
      <td>-0.004910</td>
      <td>-0.005415</td>
      <td>-0.004631</td>
      <td>NaN</td>
      <td>-0.004140</td>
      <td>-0.002925</td>
      <td>-0.003981</td>
      <td>NaN</td>
      <td>-0.003050</td>
      <td>-0.004907</td>
      <td>NaN</td>
      <td>-0.003994</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.004491</td>
    </tr>
    <tr>
      <th>2021-08-05 00:00:00.000000</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.005543</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.006422</td>
      <td>NaN</td>
      <td>-0.004897</td>
      <td>-0.005412</td>
      <td>-0.004662</td>
      <td>NaN</td>
      <td>-0.004126</td>
      <td>-0.002924</td>
      <td>-0.003975</td>
      <td>NaN</td>
      <td>-0.003100</td>
      <td>-0.004939</td>
      <td>NaN</td>
      <td>-0.004047</td>
      <td>NaN</td>
      <td>-0.003224</td>
      <td>NaN</td>
      <td>-0.004467</td>
    </tr>
    <tr>
      <th>2021-08-11 00:00:00.000000</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.005584</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.006485</td>
      <td>NaN</td>
      <td>-0.004922</td>
      <td>-0.005430</td>
      <td>-0.004804</td>
      <td>NaN</td>
      <td>-0.004316</td>
      <td>-0.003003</td>
      <td>-0.004073</td>
      <td>NaN</td>
      <td>-0.002577</td>
      <td>-0.005021</td>
      <td>NaN</td>
      <td>-0.004119</td>
      <td>NaN</td>
      <td>-0.003282</td>
      <td>NaN</td>
      <td>-0.004521</td>
    </tr>
    <tr>
      <th>2021-08-18 00:00:00.000000</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.006408</td>
      <td>NaN</td>
      <td>-0.004904</td>
      <td>-0.005420</td>
      <td>-0.004828</td>
      <td>NaN</td>
      <td>-0.004390</td>
      <td>-0.002989</td>
      <td>-0.004282</td>
      <td>NaN</td>
      <td>-0.003140</td>
      <td>-0.004485</td>
      <td>NaN</td>
      <td>-0.004109</td>
      <td>NaN</td>
      <td>-0.003276</td>
      <td>NaN</td>
      <td>-0.004505</td>
    </tr>
  </tbody>
</table>
<p>124 rows × 43 columns</p>
</div>


