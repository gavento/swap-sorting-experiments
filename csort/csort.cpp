#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <vector>
#include <algorithm>

namespace py = boost::python;

//typedef std::vector<int> int_vector;

/*
template<typename T> py::list vec_to_list(std::vector<T> &vec)
{
    py::list l;
    for (auto i: vec)
        l.append(i);
    return l;
}
*/

class RandomSort {
public:
    RandomSort(double p, int r, py::list l, int seed=42): _p(p), _r(r), _n(py::len(l)), _rnd(seed),
        _seq(py::stl_input_iterator<int>(l), py::stl_input_iterator<int>())
    { }

    RandomSort(double p, int r, int n, int seed=42): _p(p), _r(r), _n(n), _rnd(seed),
        _seq(0)
    {
        for (int i = 0; i < _n; i++)
            _seq.push_back(i);
        std::shuffle(_seq.begin(), _seq.end(), _rnd);
    }

    py::list py_get_seq()
    { 
        py::list l;
        for (auto i: _seq)
            l.append(i);
        return l;
    }

    int step()
    {
        int a = std::uniform_int_distribution<>(0, _n - 1)(_rnd);
        int b = std::uniform_int_distribution<>(std::max(a - _r, 0), std::min(a + _r, _n - 2))(_rnd);
        if (b >= a) b++;
        if (a > b)
            std::swap(a, b);
        if (!(_seq[a] < _seq[b]) == !(std::uniform_real_distribution<>(0.0, 1.0)(_rnd) < _p)) {
            std::swap(_seq[a], _seq[b]);
            return 1;
        }
        return 0;
    }

    int steps(int k)
    {
        int r = 0;
        for (int i = 0; i < k; i++)
            r += step();
        return r;
    }

    py::tuple run(int max_steps, int sampling, int conv_window, double conv_error=0.05)
    {
        std::vector<int> Is, Ws;
        int wsize = (conv_window + sampling - 1) / sampling;
        if (wsize < 8) throw PyErr_NewException("Asd", NULL, NULL);

        for (int s = 0; sampling * s < max_steps; s++) {
            Is.push_back(I());
            Ws.push_back(W());
            steps(sampling);
        }

        return py::make_tuple(Is, Ws);
    }

    int I() const
    {
        int r = 0;
        for (int a = 0; a < _n; a++) {
            for (int b = a + 1; b < _n; b++) 
                if (_seq[a] > _seq[b]) r++;
        }
        return r;
    }

    int W() const
    {
        int r = 0;
        for (int a = 0; a < _n; a++) {
            for (int b = a + 1; b < _n; b++) 
                if (_seq[a] > _seq[b]) r += _seq[a] - _seq[b];
        }
        return r;
    }

public:
    int _n, _r;
    double _p;
    std::vector<int> _seq;
    std::minstd_rand _rnd;
};

BOOST_PYTHON_MODULE(csort)
{
        using namespace boost::python;
        class_<RandomSort>("RandomSort", init<double, int, py::list>())
            .def(init<double, int, py::list, int>())
            .def(init<double, int, int>())
            .def(init<double, int, int, int>())
            .def_readonly("p", &RandomSort::_p)
            .def_readonly("r", &RandomSort::_r)
            .def_readonly("n", &RandomSort::_n)
            .def_readonly("s", &RandomSort::_seq)
            .add_property("seq", &RandomSort::py_get_seq)
            .def("step", &RandomSort::step)
            .def("steps", &RandomSort::steps)
            .def("I", &RandomSort::I)
            .def("W", &RandomSort::W)
            ;

        class_<std::vector<int>>("intVector")
            .def(vector_indexing_suite<std::vector<int>>())
            ;

}


