CC = g++
CVFLAG = -O2 -Wall -std=c++11 -I"C:\opencv\build\include" -L"C:\opencv\release\lib"
lflag = -lopencv_core2411 -lopencv_highgui2411 -lopencv_imgproc2411
target = color_transfer

$(target): $(target).cpp
	$(CC) $(target).cpp -o $(target) $(CVFLAG) $(lflag)

ALL: $(target)

clean:
	del /F $(target).exe