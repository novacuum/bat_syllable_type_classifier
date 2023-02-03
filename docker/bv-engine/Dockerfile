FROM gilleswaeber/htalpine

# graphicsmagick-dev, g++, boost-dev, linux-headers, python3-dev: All needed to install pgmagick with pip
ENV BUILD_PACKAGES="\
	graphicsmagick-dev g++ boost-dev linux-headers python3-dev\
"

# python3: python interpreter
# graphicsmagick, boost-python3: dependency of pgmagick
# sox runtime
# openjdk: for hwrecog
ENV PACKAGES="\
	python3 graphicsmagick boost-python3\
	sox\
	openjdk8-jre-base\
"

# pgmagick: imagemagick wrapper for python
ENV PIP_PACKAGES="pgmagick"

# Install Packages
RUN apk add --update --no-cache ${PACKAGES} ${BUILD_PACKAGES} &&\
# Install pip packages
	pip3 install --upgrade pip &&\
	pip3 install ${PIP_PACKAGES} &&\
# Remove build packages
	apk del ${BUILD_PACKAGES} &&\
# Add symlinks to python (like in jfloff/alpine-python)
	if [[ ! -e /usr/bin/python ]];        then ln -sf /usr/bin/python3 /usr/bin/python; fi &&\
	if [[ ! -e /usr/bin/pydoc ]];         then ln -sf /usr/bin/pydoc3 /usr/bin/pydoc; fi &&\
	if [[ ! -e /usr/bin/pip ]];           then ln -sf /usr/bin/pip3 /usr/bin/pip; fi &&\
	if [[ ! -e /usr/bin/easy_install ]]; then ln -sf $(ls /usr/bin/easy_install*) /usr/bin/easy_install; fi &&\
	echo 'Done!'

WORKDIR /root/app
