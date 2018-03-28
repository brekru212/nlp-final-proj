# composed of things i found on github/stackoverflow need to clean up
import pandas as pd
from glob import glob

import sys
import os

def parse_EPS_file(filepath):
    """ Parses a given EPS html file (does not have to have a html extension) """
    with open(filepath, 'r') as f:
        raw = f.read()

    # Trim the page to restrict to only one table
    raw_1table = raw.split('U.S. Earnings')[-1]

    # Find the HTML table
    _start = raw_1table.index('<table')
    _stop = raw_1table.index('</table>') + 8
    html_table = raw_1table[_start : _stop]

    # Parse the HTML table
    try:
        data = pd.read_html(html_table)[0]
    except:
        return None

    # Clean the table of internal headers and spaces, remove unnecessary columns
    parsed_EPS = data.loc[~data.ix[:, 1].isin([pd.np.nan, 'Symbol']), data.columns[:5]]
    parsed_EPS.columns = ['Company', 'Symbol', 'Surprise', 'Reported_EPS', 'Consensus_EPS']

    # Add date based on the filename
    parsed_EPS['Date'] = pd.to_datetime(filepath.split('/')[-1].split('.')[0], format='%Y%m%d')

    return parsed_EPS

def parse_files(input_folder, output_folder):
    all_files = glob('{}*.txt'.format(input_folder))

    FAILED = []
    EMPTY = []
    count = 0

    MAJOR = max(len(all_files) // 10, 1)
    MINOR = max(len(all_files) // 100, 1)

    for filename in all_files:

        # Get output path
        outpath = '{fld}{f}.csv'.format(fld=output_folder, f=filename.split('/')[-1].split('.')[0])

        try:
            data = parse_EPS_file(filename)
        except:
            print('error=' + filename + '\n')
            FAILED.append(filename)
        else:
            if data is None:
                EMPTY.append(filename)
                print('end=' + filename +'\n')
            else:
                # Save the data
                with open(outpath, 'w') as f:
                    data.to_csv(f)

        # Update counter and print processing mark
        count += 1


    print('\n\n... Done!')

    print('Processed {} files, of which:'.format(count))
    print('\033[31m  - Failed:', len(FAILED), '\033[0m')
    print('\033[33m  - Empty:', len(EMPTY), '\033[0m\n')

parse_files('EPS/', 'parsed-EPS/')