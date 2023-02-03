# Bat syllable type classifier
The project is hosted at https://github.com/simon-at-fugu/bat_syllable_type_classifier.git
It is based on The *Bird Voice Server* bachelor project of Gilles Waeber, student at the *Haute École d'Ingénierie et d'Architecture de Fribourg (HEIA-FR)*, realized at the *Human Interface Laboratory* in *Kyushu University*.
This project contains the Software use for the master thesis *Automated classification of syllable types in vocal sequences of pups of the greater sac-winged bat Saccopteryx bilineata* by Simon Hiller

## What this is about
The aim of this project is to investigate the possibilities and limitations of using deep neural networks to classify bat syllable types of pups of the greater sac-winged bat *Saccopteryx bilineata*.

## Structure
- `data` contains the data, as separate git projects and results
- `experiments` playground based on notebooks
- `src` contains the different softwares
- `docker` contains docker files from *Bird Voice Server*
- `bin` batch files used for UBELIX

## Installation
Requirements:
- Conda: download at [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.8 or newer: download at [python.org](https://www.python.org/).
- SoX (*Sound eXchange): download at [sox.sourceforge.net](http://sox.sourceforge.net/).

The HWRecog software was used for HoG extraction but is now replaced by OpenCV

- Download and install miniconda (Python 3) from https://docs.conda.io/en/latest/miniconda.html (or using scoop)
- Download and install SoX from http://sox.sourceforge.net/ (or using apt/scoop)
- Download and install Java 8+, e.g. from https://www.java.com/download/ (or using apt/scoop)

```sh
sudo apt install sox default-jre  # Ubuntu (conda has to be install manually)
scoop install miniconda3 sox openjdk  # Windows

conda create -n bat_syllable_type_classifier  # create environment
conda activate bat_syllable_type_classifier  # switch to birdvoice environment
conda install numpy scipy  # required: computation
conda install tqdm filelock mutagen notebook nbconvert # required: utilities
pip install tensorflow-gpu librosa wavinfo tensorboard # ANN framework & audio tools
conda install opencv matplotlib seaborn scikit-learn opencv # optional: for graphs

conda install scikit-image # optional: for heatmaps only
# install innvestigate
pip install git+https://github.com/simon-at-fugu/innvestigate.git@updates_towards_tf2.0


conda develop path/to/src  # add source directory to the environment path (not need for ubelix)
```

### Install SOX on UBELIX
```sh
cd ~/dl/

wget http://downloads.sourceforge.net/libpng/libpng-1.6.16.tar.xz
tar xf libpng-1.6.16.tar.xz
cd ~/dl/libpng-1.6.16/
./configure --prefix=$HOME/app/libpng-1.6.16
make -s && make install

cd ~/dl/
wget https://nchc.dl.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.gz
tar xvfz sox-14.4.2.tar.gz
cd ~/dl/sox-14.4.2/
./configure LDFLAGS="-L$HOME/app/libpng-1.6.16/lib" CPPFLAGS="-I$HOME/app/libpng-1.6.16/include" --prefix=$HOME/app/sox-14.4.2
```

### Datasets
The datasets without the audio files are public available:
- simple call test: https://github.com/simon-at-fugu/simple_call_test
- simple call sequence: https://github.com/simon-at-fugu/simple_call_seq

## Run experiments
On the ubelix you can use the shell files in ./bin/.
First init the environment with:
```sh
. load_ubelix_env.sh
```
Then call one of these batch files:
- *sbatch_sct.sh*: execute all simple call test experiments
- *sbatch_scs.sh*: execute all simple call sequence experiments
- *sbatch job_notebookrunner.sh*: execute notebook base report generation for all results (notebook location is: ./data/results/report/)
- *sbatch job_scs_predict.sh*: execute prediction job

Otherwise, you can start the test experiments as follows:
### Test experiments
Run one of the simple call test experiments:
```sh
$PROJECT_DIR = 'path/to/this/project/'
export BSC_DATASET_NAME='simple_call_test' #define which dataset to use
cd $PROJECT_DIR/src/
pyhton scripts/sct.py -v=$variant -i=$index
```

Where:
- $variant: name of the test experiment (e.g. padding)
- $index: index of the configuration from 0 to 3

### Sequence experiments
Run one of the simple call sequence experiments:
```sh
$PROJECT_DIR = 'path/to/this/project/'
export BSC_DATASET_NAME='simple_call_seq'  #define which dataset to use
cd $PROJECT_DIR/src/
pyhton scripts/scs.py -v=r3 -i=$index
```

Where:
- $index: index of the configuration from 0 to 3

### Generate result books (notebooks)

```sh
$PROJECT_DIR = 'path/to/this/project/'
rm $PROJECT_DIR/data/results/report/*.html #delete already generated files (not needed)
cd $PROJECT_DIR/src/
pyhton scripts/notebookrunner.py
```

# Results
The experiment specific data like the stored model are located in the dataset folder under `data/`
The experiment results files and notebook report files are located in the folder `data/results/`

