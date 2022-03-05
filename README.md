# Deep Learning from Synthetic Datafor Atomic Force Microscopy Image Reconstruction

This is code for the paper submitted to the journal Ultramicroscopy. The pre-print is available at: TBA.

## Setup

First download the whole repo:

    git clone https://github.com/kocurvik/gefsem

Next, create the conda environment using the following command:

    conda env create -f environment.yml
You can then activate the environment using:

    conda activate gefsem
You will also need to add the main folder to PYTHONPATH:

    export PYTHONPATH=/path/to/the/repo/gefsem;
## Running the evaluation scripts
You need to first download the evaluation files along with the json files containing the annotations for alignment. They are stored on [OneDrive](TBA).

You will also need to download the pytorch model form [OneDrive]([resunet.pth](https://liveuniba-my.sharepoint.com/:u:/g/personal/kocur15_uniba_sk/Ec_Pg042vzJDsnS5WmnAWWIBV92nvYivkoORr5ogPcPbNA?e=bbB0Es)).

You can then run the evaluation and generate a pdf with results (you might need to create a vis folder in you repo directory) by running:

    python eval/run_eval.py -l -e -i -nw 6 resunet.pth path/to/downloaded/EvalData
You can run all of the baselines by running:

    python eval/run_eval.py -l -e -i  -nw 6 baseline path/to/downloaded/EvalData
    python eval/run_eval.py -l -e -i -nw 6  -g baseline path/to/downloaded/EvalData
    python eval/run_eval.py -l -e -i -nw 6  -a baseline path/to/downloaded/EvalData
    python eval/run_eval.py -l -e -i -nw 6  -m baseline path/to/downloaded/EvalData

You can omit or change the -nw 6 argument to change the number of workers to fit your machine. If you have issues with multithreading then you can use -nw 1.

Afterwards you can generate a table by running:

    python eval/generate_table.py path/to/downloaded/EvalData
## Training
If you want to train the network you will have to follow these steps:
### Generating the dataset
You will need the pickle file with a dataset of real tips from [OneDrive](https://liveuniba-my.sharepoint.com/:u:/g/personal/kocur15_uniba_sk/EWDE0kbGQBJNr173VGjKLB0BlEj6PYQFJ_YOyzfFX1ZzHQ?e=gyhP3T)
You need to generate the validation dataset first:

    python synth/synthetizer.py -n 3000 -nw 6 -s val -aa /path/to/dowloaded/tips.pkl /path/where/you/want/dataset
You can check out other args to tweak the parameters of the generated dataset.
### Training your own model
Training data will be generated on the fly based on the same parameters which were used for generating the validation dataset. The hashed part of the dataset may vary so check out where the validation was generated.
Run the training:

    python network/train.py -b 16 -o adam -exp 0 -e 10 -de 1 -lr 1e-3 -wd 1e-2 -nw 6 -g 0 /path/where/you/want/dataset/555e6565715b5fd76a38f56c7fbf2098ab3e69a3/
You can change the parameters. Especially the batch size and number of epochs. Check the help. If you want to train with different training steps use the `-r` option and the `-expr` option which starts from a previous experiment determined by the `-exp` option. For example:

    python network/train.py -b 16 -o adam -exp 0 -e 10 -de 1 -lr 1e-3 -wd 1e-2 -nw 6 -g 0 /path/where/you/want/dataset/555e6565715b5fd76a38f56c7fbf2098ab3e69a3/
    python network/train.py -b 16 -o adam -exp 1 -expr 0 -r 9 -e 20 -de 1 -lr 1e-4 -wd 1e-2 -nw 6 -g 0 /path/where/you/want/dataset/555e6565715b5fd76a38f56c7fbf2098ab3e69a3/
    python network/train.py -b 16 -o adam -exp 2 -expr 1 -r 19 -e 30 -de 1 -lr 1e-5 -wd 1e-2 -nw 6 -g 0 /path/where/you/want/dataset/555e6565715b5fd76a38f56c7fbf2098ab3e69a3/
The model files for every checkpoint will appear in the `checkpoints/555e6565715b5fd76a38f56c7fbf2098ab3e69a3/expnum` folder. You can rename them for evaluation.

### Creating your own annotated keypoints
You may also want to create your own keypoints. You can use the annotator for this purpose.

    python eval/annotator.py path/to/folder/with/gwy/files
You will have to select and click on the keypoints in the first image. Then you will have to click on the same keypoint in the rest of the images. You need at least 3 keypoints, but 4-5 is better. You can click any key on the keyboard to move onto the next image.

You can use the `-m` option to run manual alignment of left-to-right and right-to-left scans if the simple MSE method does not work. If you do this you will need to manually align the images. You can use the keyboard controls: t and v control contrast, k and s control the offset, c continues to next image and saves offset.

This should create a json file which can be used in the validation script (as discussed above).
### Running on gwy files
TBA
