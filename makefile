CC=g++
STD=-std=c++11
WARN=-Wall -Wextra -Wpedantic -Wformat-security -Wfloat-equal -Wshadow\
     -Wconversion -Winline -Wpadded
OPT=-O2 -march=native -ffinite-math-only -fno-signed-zeros
DBG=-O0 -g -ggdb -DDEBUG=2
EXTRA=$(shell pkg-config --cflags opencv) -DINVERT_AXIS
LINK=$(shell pkg-config --libs opencv)
OPENCV_PATH=/usr/local/Cellar/opencv/3.4.3/share/OpenCV

DEBUG=$(STD) $(WARN) $(DBG) $(EXTRA) $(LINK)
RELEASE=$(STD) $(WARN) $(OPT) $(EXTRA) $(LINK)

debug:
	ln -s $(OPENCV_PATH)/haarcascades/* .
	g++ ifocus.cpp $(DEBUG) -o ifocus
release:
	ln -s $(OPENCV_PATH)/haarcascades/* .
	g++ ifocus.cpp $(RELEASE) -o ifocus
clean:
	rm -f ifocus
	find . -type l -delete
