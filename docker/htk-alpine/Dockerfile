FROM alpine:latest

# Download at http://htk.eng.cam.ac.uk/ftp/software/HTK-3.4.1.tar.gz
COPY HTK-3.4.1.tar.gz ./HTK/
COPY HResults.patch ./HTK/

ENV BUILD_PACKAGES="g++ make linux-headers"

ENV PACKAGES=""

# Default workdir: /
RUN	cd /HTK &&\
# Load packages required for compilation
	apk add --update --no-cache ${BUILD_PACKAGES} ${PACKAGES} &&\
# Compile
	tar -xzf HTK-3.4.1.tar.gz &&\
	cd htk &&\
# The original Makefile has a bug (8 spaces instead of a TAB)
	sed -i -r 's/ {8}/\t/g' ./HLMTools/Makefile.in &&\
# Patch HResults.c
	patch ./HTKTools/HResults.c /HTK/HResults.patch &&\
# The original configure forces 32-bit compilation but alpine has no 32-bit support
	sed -i -r 's/-m32/-m64/g' ./configure ./configure.ac &&\
	./configure --disable-hslab &&\
	make all &&\
	make install &&\
# Remove sources and packages not needed anymore
	cd / &&\
	rm -rf HTK &&\
	apk del ${BUILD_PACKAGES} &&\
	echo 'Done !'

WORKDIR /root
