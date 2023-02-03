BirdVoice Engine
============

The Ruby software has been entirely rewritten in Python (including the weird perl script).

The new software is built on series of operations that can be chained. Each chain of operations has an unique and reproducable result. Parts of the chain can be reused if included in multiple chains.

Prerequisites
-------------
Requirements:
- Python 3.6 or newer: download at [python.org](https://www.python.org/).
- The *pgmagick* python package: download using `pip install pgmagick` or `apt install python-pgmagick` or download at [lfd.uci.edu/~gohlke/pythonlibs](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pgmagick) for Windows
- SoX (*Sound eXchange): download at [sox.sourceforge.net](http://sox.sourceforge.net/).
- The HTK Speech Recognition Toolkit: download the sources at [htk.eng.cam.ac.uk](http://htk.eng.cam.ac.uk/).

Alternatively, use docker:
- Get docker at [docker.com](https://www.docker.com/)
- Use the `gilleswaeber/htkbvhog` image

The HWRecog software is required for HoG extraction but is bundled (also bundles OpenCV).

Quick start
-----------
First, load the data using the load script located in the data folder:
```sh
# Linux
data/load.sh
# Windows (as admin)
data/load.ps1
```
Then go to the *spectrogram-htk* folder and run one of the experiments:
```sh
# Starting at project root
cd software/spectrogram-htk
python -m experiments.jungle-basics
```

Artifacts
---------
All collections are immutable.

Type           | Description
-------------- | -----------
folder         | a folder located in `data`
audio          | a collection of audio tracks
audio_matching | a collection of audio tracks that will reproduce the pipeline of another audio collection
conf           | a list of samples
profile        | a collection of noise profiles
spectrogram    | a collection of spectrograms
spectrogram_b  | a collection of spectrograms, binarized
features       | a collection of extracted features
blank_model    | a hmm model with nothing on it, linked to a collection of features
model          | a trained hmm model, linked to a collection of features
results        | the results for a specific pipeline


Operations
----------

Function | Input | Output | Description
-------- | ----- | ------ | -----------
`load_audio`(*folder*) | - | audio | take audio files from a folder
`load_audio`(*folder*, *conf*) | - | audio | take audio files from a folder and filter them using a configuration file
`load_audio_matching`(*folder*) | - | audio | take audio files from a folder
`load_audio_matching`(*folder*, *conf*) | - | audio | take audio files from a folder and filter them using a configuration file
`profile`(*folder*) | - | profile | take noise profiles from a folder
*number* | - | number | a single number, integer
`irange`(*first*, *last*) | - | number | a number range, inclusive
`irange`(*first*, *last*, *step*) | - | number | a number range, inclusive
`merge_channels`() | audio | audio | merge the audio channels for each sample
`noise_reduce`(*noise*\|*audio*, *sensitivity*) | audio | audio | apply a noise reduction, *noise* can be either a collection of audio files (can be *audio_matching*) or of noise profiles,*sensitivity*, in %, is the noise threshold used for the reduction
`geo_features`(*thr*) | audio | features | extract the features from the spectrogram of the audio files, *thr*, in %, is the threshold used when deciding if a pixel is black or white
`create_model`(*states*, *mixtures*) | features | model | create an HMM model with a given number of states and mixtures
`recognize`(*audio_matching*) | model | result | extract recognition results for the given model (the audio collection passed will go through the same preprocessing pipeline as the one used for the model)

Examples
--------
Below are some proposed pipelines.

Pipeline for the thesis of F. Ziegler:
```python
for states in irange(5, 40, 5) + [50]:
	for mixtures in [1,3,5,7,10,13,15,20,25]:
		# Start with the training data, no channel merging
		(load_audio('jungle', 'training')
		# Noise reduction with sensitivity 0.21
			.noise_reduce(load_audio('jungle_noise'),21)
		# Extract features with threshold of 50%
			.geo_features(50)
		# Test with 5,10,15,20,25,30,35,40 and 50 states and various amounts of gaussian mixtures
			.create_model(states, mixtures)
		# Apply on validation data
			.recognize(load_audio_matching('jungle', 'validate'))
		# Extract the results
            .extract('jungle')
        # Run the pipeline
			.run())
```

Better pipeline (98.22% accuracy):
```python
# Start with the training data, channel merging
results = (load_audio('jungle', 'training').merge_channels()
# Noise reduction with sensitivity 0.21
    .noise_reduce(load_audio_matching('jungle_noise'),21)
# Extract features with threshold of 55%
    .geo_features(55)
# Test with 15 and 15 gaussian mixtures
    .create_model(15,15)
# Apply on validation data
    .recognize(load_audio_matching('jungle', 'validate'))
# Extract the results
    .extract('jungle')
# Run the pipeline
	.run())
```
