

all: clean run

run:
	c++  -O2 \
		TB_LayerTop.cpp LayerTop.cpp

clean:
	rm -f conv2.bin

