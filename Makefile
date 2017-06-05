

all: clean run

run:
	c++  -O2 \
		TB_LayerTop.cpp LayerTop.cpp caffe.pb.cc -lprotobuf -lpthread

clean:
	rm -f conv2.bin

