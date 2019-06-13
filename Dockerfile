FROM ubuntu:18.04
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y python3.6 python3-pip --no-install-recommends
RUN apt-get install -y --no-install-recommends ffmpeg espeak

RUN pip3 install setuptools
RUN pip3 install numpy==1.16.3
RUN pip3 install pytest==4.4.1
RUN pip3 install SoundFile==0.10.2
RUN pip3 install tensorflow==1.13.1
RUN pip3 install gTTS==2.0.3
RUN pip3 install matplotlib==3.0.3
RUN pip3 install pydub==0.23.1
RUN pip3 install mypy==0.701

RUN apt-get install -y --no-install-recommends make

