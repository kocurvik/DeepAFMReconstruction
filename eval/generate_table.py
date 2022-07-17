import argparse
import json
import os

import numpy as np

DIRNAME_DICT = {'d008': 'Wafers', 'D010_Bunky': 'Cells', 'INCHAR (MFM sample)': 'Permalloy', 'Kremik': 'Silicon',
                'loga': 'Logos', 'Neno': 'Neno', 'Tescan sample': 'Patterns', 'TGQ1': 'TGQ1', 'TGZ3': 'TGZ3'}

def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to the output of the run_eval.py file')
    args = parser.parse_args()
    return args


def generate_table(data_path):
    # Generates tables from the paper
    json_paths = [x for x in os.listdir(data_path) if '.json' in x]
    all_results = {}
    for json_path in json_paths:
        model_name = json_path.split('_results')[0]
        with open(os.path.join(data_path, json_path), 'r') as f:
            all_results[model_name] = json.load(f)
            all_results[model_name] = sorted(all_results[model_name], key=lambda x: x['dir'])

    for i in range(len(all_results[model_name])):
        dir = all_results[model_name][i]['dir']
        print("*" * 20)
        print("DIR: ", dir)
        for model, list_of_results in all_results.items():
            print("Model: ", model)
            rmse = np.array(list_of_results[i]['rmse'])
            mse = np.array(list_of_results[i]['mse'])
            rcorr = np.array(list_of_results[i]['rcorrelation'])
            corr = np.array(list_of_results[i]['correlation'])
            rmse = rmse[np.triu_indices(len(rmse), k=1)]
            mse = mse[np.triu_indices(len(mse), k=1)]

            print("rmse - mean: {} \t median: {}, \t rcorr: - mean {} \t median: {}".format(np.mean(rmse), np.median(rmse), np.mean(rcorr), np.median(rcorr)))
            # print("mse - mean: {} \t median: {}".format(np.mean(mse), np.median(mse)))

    dirs = [all_results[model_name][i]['dir'] for i in range(len(all_results[model_name]))]
    dir_names = [DIRNAME_DICT[d] for d in dirs]
    for metric in ['rmse', 'rcorrelation']:
        for statistic in ['Mean', 'Median']:
            print(20 * '*')
            print(metric + ' + ' + statistic)
            print(20 * '*')
            topline = 'Model & ' + ' & '.join(dir_names) + '\\\\ \\hline'
            print(topline)
            for model, list_of_results in all_results.items():
                vals = []
                for i in range(len(all_results[model_name])):
                    val = np.array(list_of_results[i][metric])
                    val = val[np.triu_indices(len(val), k=1)]
                    if metric == 'rmse':
                        val = val / 1e-9
                    if statistic == 'Mean':
                        val = '{:.4f}'.format(np.mean(val))
                    else:
                        val = '{:.4f}'.format(np.median(val))
                    # if 'baseline' not in model:
                    #     val = '\\textbf{' + val + '}'

                    vals.append(val)

                line = model + (30 - len(model))*' ' + '& ' + ' & '.join(vals) + '\\\\ \\hline'
                print(line)


if __name__ == '__main__':
    args = parse_command_line()
    generate_table(args.data_path)

