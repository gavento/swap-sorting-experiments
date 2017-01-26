all: csort.so test

csort.so: csort.cpp Makefile
	g++ -o csort.so -shared -fPIC -g -O3 -std=c++11 -Wall $< \
	    -ldl -lpthread -lutil -lboost_python-py35 `pkg-config python3 --cflags --libs`

clean:
	rm -f csort.so csort.o

test: csort.so
	python3 -c "import csort; rs=csort.RandomSort(0.1, 1, 100); rs.steps(200); assert rs.n == 100; assert len(rs.seq) == rs.n; assert len(rs.Is) == len(rs.Ws);"
	python3 -c "import csort; rs=csort.RandomSort(0.1, 100, 100); t = rs.converge_on_I(-1, 0.05); assert rs.I_stab() < 1500.0; assert t > 300 and t < 600;"

.PHONY: all clean test
