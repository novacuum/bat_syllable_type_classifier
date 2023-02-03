BirdVoice Engine
================

Description
-----------
This docker image provides the toolset necessary for the bird voice recognition project, namely the *HTK Toolkit*, *python3*, *SoX* and the *pgmagick* library.

Building
--------
First build the **HTK-Alpine** image in the folder *htk-alpine*, used as a base image for this one.

Then run:
```sh
docker build --rm -t gilleswaeber/bv-engine .
```

Running
-------
To get a simple shell (ash) run:
```sh
docker run --rm -ti gilleswaeber/bv-engine
```

To mount a local folder to /root/myproject in the container run:
```sh
# Linux
docker run --rm -ti -v $(pwd):/root/myproject gilleswaeber/bv-engine python project.py
# Windows (PowerShell)
docker run --rm -ti -v ${pwd}:/root/myproject gilleswaeber/bv-engine python project.py
```
