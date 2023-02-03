FROM gilleswaeber/htkbvhog

# Needed for pip install
ENV BUILD_PACKAGES="\
	gcc linux-headers musl-dev python3-dev\
"

ENV PACKAGES="\
"

ENV PIP_PACKAGES="\
    channels\
    Django\
    djangorestframework\
"

# Install Packages
RUN apk add --update --no-cache ${PACKAGES} ${BUILD_PACKAGES} &&\
# Install pip packages
	pip3 install --upgrade pip &&\
	pip3 install ${PIP_PACKAGES} &&\
# Remove build packages
	apk del ${BUILD_PACKAGES} &&\
# Done
	echo 'Done!'

WORKDIR /root/app
