all: csort.so test

csort.so: csort.cpp Makefile
	g++ -o csort.so -shared -fPIC -g -O0 -std=c++11 $< \
	    -ldl -lpthread -lutil -lboost_python-py35 `pkg-config python3 --cflags --libs`

clean:
	rm -f csort.so csort.o

test: csort.so
	python3 -c "import csort; rs=csort.RandomSort(0.1, 1, [6, 3, 4, 8, 7, 1]); assert rs.n == 6;"

.PHONY: all clean test
