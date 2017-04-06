all: csort.so test plot

csort.so: csort.cpp Makefile
	@echo "---- Compiling ----"
	g++ -o csort.so -shared -fPIC -g -O3 -std=c++11 -Wall $< \
	    -ldl -lpthread -lutil -lboost_python-py35 `pkg-config python3 --cflags --libs`

clean:
	@echo "---- Cleanup ----"
	rm -f csort.so csort.o

test: csort.so
	@echo "---- Running basic tests ----"
	python3 -c "import csort; rs=csort.RandomSort(0.1, 1, 100); rs.steps(200); assert rs.n == 100; assert len(rs.seq) == rs.n; assert len(rs.Is) == len(rs.Ws);"
	python3 -c "import csort; rs=csort.RandomSort(0.1, 100, 100); t = rs.converge_on_I(0.05, -1); assert rs.I_stab() < 1500.0; assert t > 300 and t < 600;"
	python3 -c "import csort; rs=csort.RandomSort(0.1, 100, 100); t = rs.converge_on_W(0.05, -1); assert rs.W_stab() < 32000.0; assert t > 300 and t < 600;"

plot: csort.so
	@echo "---- Running and plotting experiments ----"
	python3 plotting.py	

.PHONY: all clean test plot
