

all: clean run

run:
	g++ -g \
		./ip_tb.cpp ./ip.cpp ./caffe.pb.cc -lprotobuf -lpthread

clean:
	rm -f conv2.bin

