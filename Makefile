

all:
	g++ -g \
		./ip_tb.cpp ./ip.cpp ./caffe.pb.cc -lprotobuf -lpthread
