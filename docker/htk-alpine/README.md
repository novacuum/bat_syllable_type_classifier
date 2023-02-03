HTK-Alpine
==========

Description
-----------
This is a lightweight docker image for the HTK Speech Recognition Toolkit 3.4.1 on alpine linux.
This image is less than 70MB.

HTK is patched just before compilation:
- Confusion matrix size limit is dropped in HResults
- Confusion matrix format has been fixed (hardcorded field size)
- Binaries are compiled in 64-bit rather than 32-bit
- Makefile is fixed

Building
--------
Requires `HTK-3.4.1.tar.gz` that can be downloaded at http://htk.eng.cam.ac.uk/download.shtml

```sh
docker build --rm -t gilleswaeber/htalpine -t gilleswaeber/htalpine:3.4.1b .
```

Note that compilation will generate a bunch of warnings.

Running
-------
To get a simple shell (ash):
```sh
docker run --rm -ti htalpine
```

To mount a local folder to /root/myproject in the container run:
```sh
# Linux
docker run --rm -ti -v $(pwd):/root/myproject htalpine
# Windows (PowerShell)
docker run --rm -ti -v ${pwd}:/root/myproject htalpine
```

To run HInit:
```sh
docker run --rm htalpine HInit
```
