import argparse
import json
import os

import numpy as np

def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    args = parser.parse_args()
    return args


def generate_table(data_path):
    json_paths = [x for x in os.listdir(data_path) if '.json' in x]
    all_results = {}
    for json_path in json_paths:
        model_name = json_path.split('.')[0]
        with open(os.path.join(data_path, json_path), 'r') as f:
            all_results[model_name] = json.load(f)

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



if __name__ == '__main__':
    args = parse_command_line()
    generate_table(args.data_path)

