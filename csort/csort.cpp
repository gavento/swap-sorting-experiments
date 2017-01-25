#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/format.hpp>
#include <vector>
#include <algorithm>
#include <numeric>

namespace py = boost::python;

class RandomSort {
public:
    RandomSort(double p, int r, std::vector<int> const &seq, int sampling=-1, int seed=42):
        _p(p), _r(r), _n(seq.size()), _rnd(seed), _Is(), _Ws(), _Ts(), _T(0),
        _sampling(sampling > 0 ? sampling : std::max((int)(ET() / 1000), 1))
    { }

    RandomSort(double p, int r, py::list l, int sampling=-1, int seed=42):
        RandomSort(p, r, std::vector<int>(py::stl_input_iterator<int>(l), py::stl_input_iterator<int>()), sampling, seed)
    { }

    RandomSort(double p, int r, int n, int sampling=-1, int seed=42): 
        RandomSort(p, r, std::vector<int>(n), sampling, seed)
    {
        for (int i = 0; i < _n; i++)
            _seq.push_back(i);
        std::shuffle(_seq.begin(), _seq.end(), _rnd);
    }

    double ET() const
    {
        return 1.0 * _n * _n / _r / (0.5 - _p) / 2;
    }

    int steps(int k)
    {
        if ((_sampling > 0) && (_T == 0) && (_Is.size() == 0)) {
            _Is.push_back(I());
            _Ws.push_back(W());
            _Ts.push_back(_T);
        }

        int r = 0;
        for (int i = 0; i < k; i++) {
            r += step();
            _T++;
            if ((_sampling > 0) && (_T % _sampling == 0)) {
                _Is.push_back(I());
                _Ws.push_back(W());
                _Ts.push_back(_T);
            }
        }
        return r;
    }

    int run(int max_steps, int conv_window=-1, double conv_error=0.05, bool conv_on_I=true)
    {
        int wsize = 0;
        if (conv_window > 0) {
            wsize = (conv_window + _sampling - 1) / _sampling;
            if (wsize < 8) {
                fprintf(stderr, "WARNING: wsize only %d (with n=%d, ET=%d, conv_window=%d, sampling=%d)\n",
                        wsize, _n, (int)(ET()), conv_window, _sampling);
            }
        }

        for (int s = 0; _T < max_steps; s++) {
            steps(_sampling);
            if ((conv_window > 0) && (_T > 2 * wsize * _sampling)) {
                auto end = conv_on_I ? _Is.end() : _Ws.end();
                double mean1 = std::accumulate(end - 2 * wsize, end - 1 * wsize, 0.0) / (double) wsize;
                double mean2 = std::accumulate(end - 1 * wsize, end - 0 * wsize, 0.0) / (double) wsize;
                if (mean1 < mean2 * (1.0 + conv_error))
                    return _T - (3 * wsize / 2);
            }
        }

        return -1;
    }

    int run_conv(bool conv_on_I=true)
    {
        return run(ET() * 10, std::max((int)(ET() / 50), 8), 0.0, conv_on_I);
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

    std::string info_str() const
    {
        return boost::str( boost::format("<RandomSort(%f, %d, %d) @T=%d>") % _p % _r % _n % _T );
    }

public:
    int _n, _r, _sampling, _T;
    double _p;
    std::minstd_rand _rnd;
    std::vector<int> _seq;
    std::vector<int> _Is, _Ws, _Ts;

    int step()
    {
        if (_n <= 1) return 0;
        int a = std::uniform_int_distribution<>(0, _n - 1)(_rnd);
        int b = std::uniform_int_distribution<>(std::max(a - _r, 0), std::min(a + _r, _n - 2))(_rnd);
        if (b >= a) {
            b++;
        } else {
            std::swap(a, b);
        }
        assert((0 <= a) && (a < b) && (b < _n));
        if (!(_seq[a] < _seq[b]) == !(std::uniform_real_distribution<>(0.0, 1.0)(_rnd) < _p)) {
            std::swap(_seq[a], _seq[b]);
            return 1;
        }
        return 0;
    }
};

BOOST_PYTHON_MODULE(csort)
{
        using namespace boost::python;
        class_<RandomSort>("RandomSort", init<double, int, py::list>())
            .def(init<double, int, py::list, int>())
            .def(init<double, int, py::list, int, int>())
            .def(init<double, int, int>())
            .def(init<double, int, int, int>())
            .def(init<double, int, int, int, int>())
            .def_readonly("p", &RandomSort::_p)
            .def_readonly("r", &RandomSort::_r)
            .def_readonly("n", &RandomSort::_n)
            .def_readonly("seq", &RandomSort::_seq)
            .def_readonly("sampling", &RandomSort::_sampling)
            .def_readonly("T", &RandomSort::_T)
            .def_readonly("Is", &RandomSort::_Is)
            .def_readonly("Ws", &RandomSort::_Ws)
            .def_readonly("Ts", &RandomSort::_Ts)
            .def("steps", &RandomSort::steps)
            .def("run", &RandomSort::run)
            .def("run_conv", &RandomSort::run_conv)
            .def("ET", &RandomSort::ET)
            .def("I", &RandomSort::I)
            .def("W", &RandomSort::W)
            .def("__str__", &RandomSort::info_str)
            .def("__repr__", &RandomSort::info_str)
            ;

        class_<std::vector<int>>("intVector")
            .def(vector_indexing_suite<std::vector<int>>())
            ;

}


