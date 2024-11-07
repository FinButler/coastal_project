import pandas as pd
import numpy as np
from pyextremes import EVA
from pyextremes import get_extremes
import matplotlib.pyplot as plt
from pyextremes import plot_parameter_stability
from pyextremes import plot_return_value_stability
from pyextremes import plot_threshold_stability

def main():
    data_path = 'project_data/'
    dates = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

    all_data = pd.DataFrame()

    for date in dates:
        # Load the file, dropping rows with missing or invalid wave heights
        df = pd.read_csv(
            data_path + '44095h' + str(date) + '.txt',
            sep=r'\s+',
            header=0,
            skiprows=[1],
            usecols=['#YY', 'MM', 'DD', 'hh', 'mm', 'WVHT']
        )
        df = df.dropna(subset=['WVHT'])
        df = df[df['WVHT'] != 99]

        # Create a datetime column
        df['Datetime'] = pd.to_datetime(df['#YY'].astype(str) + '-' +
                                        df['MM'].astype(str).str.zfill(2) + '-' +
                                        df['DD'].astype(str).str.zfill(2) + ' ' +
                                        df['hh'].astype(str).str.zfill(2) + ':' +
                                        df['mm'].astype(str).str.zfill(2))

        all_data = pd.concat([all_data, df[['Datetime', 'WVHT']]])

    # Reset the index of the combined DataFrame and set Datetime as index
    all_data.reset_index(drop=True, inplace=True)
    all_data.set_index('Datetime', inplace=True)
    wave_heights = all_data['WVHT']

    model = EVA(wave_heights)
    model.get_extremes("POT", threshold=3, r="24h")

    model.plot_extremes(show_clusters=True)

    plot_parameter_stability(wave_heights)

    plot_return_value_stability(
        wave_heights,
        return_period=100,
        thresholds=np.linspace(3, 6.8, 20),
        alpha=0.95,
    )

    plot_threshold_stability(
        wave_heights,
        return_period=100,
        thresholds=np.linspace(3, 6.8, 20),
    )

    model.fit_model()
    model.plot_diagnostic(alpha=0.95)

    plt.show()

if __name__ == '__main__':
    main()
