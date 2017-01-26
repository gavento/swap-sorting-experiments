#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/format.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace py = boost::python;

class RandomSort {
public:
    RandomSort(double p, int r, std::vector<int> const &seq, int sampling=-1, int seed=42):
        _p(p), _r(r), _n(seq.size()), _rnd(seed), _Is(), _Ws(), _Ts(), _T(0),
        _sampling(sampling > 0 ? sampling : std::max((int)(ET() / 1000), 1)), _seq(seq)
    {
        if (_r > _n)
            _r = _n;
        assert((_n >= 1) && (_r >= 1) && (_p >= 0.0) && (_p <= 1.0));
    }

    RandomSort(double p, int r, py::list l, int sampling=-1, int seed=42):
        RandomSort(p, r, std::vector<int>(py::stl_input_iterator<int>(l), py::stl_input_iterator<int>()), sampling, seed)
    { }

    RandomSort(double p, int r, int n, int sampling=-1, int seed=42): 
        RandomSort(p, r, std::vector<int>(n), sampling, seed)
    {
        for (int i = 0; i < _n; i++)
            _seq[i] = i;
        std::shuffle(_seq.begin(), _seq.end(), _rnd);
    }

    double ET() const
    {
        return 0.5 * (1 + 0.25 * std::log2(_r)) * _n * _n / _r / (0.5 - _p);
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

    int converge_on_I(int conv_window=-1, double rel_error=0.05)
    {
        if (conv_window <= 0) {
            conv_window = (int)(4 * ET() * rel_error);
        }
        int win_samples = std::max(conv_window / _sampling, 8);
        if (win_samples <= 8) {
            fprintf(stderr, "WARNING: win_samples only %d (with n=%d, ET=%d, conv_window=%d, rel_error=%f, sampling=%d)\n",
                    win_samples, _n, (int)(ET()), conv_window, rel_error, _sampling);
        }

        while (true) {
            steps(_sampling);
            int Ttest = _T / 2 - 10000 - conv_window;
            if (Ttest > 0) {
                int Stest = Ttest / _sampling;
                double mean_window = std::accumulate(_Is.begin() + Stest, _Is.begin() + Stest + win_samples, 0.0) / (double) win_samples;
                double mean_stable = I_stab();
                if (mean_window <= mean_stable * (1.0 + rel_error)) {
/*                    fprintf(stderr, "DEBUG: Converged with T=%d, Ttest=%d, Stest=%d, Ewindow=%f, Estable=%f.\n",
                            _T, Ttest, Stest, mean_window, mean_stable);
*/                    return Ttest;
                }
            }
            if (_T > ET() * 20 + 1000000) {
                fprintf(stderr, "FATAL: more that 20*E[T] steps taken. Aborting. (with p=%f, r=%d, n=%d, ET=%d, T=%d, conv_window=%d, rel_error=%f, sampling=%d)\n",
                        _p, _r, _n, (int)(ET()), _T, conv_window, rel_error, _sampling);
                throw NULL;
            }
        }
    }

    double I_stab() const
    {
        int spls = _Is.size() / 4;
        return std::accumulate(_Is.end() - spls, _Is.end(), 0.0) / (double)(spls);
    }

    double W_stab() const
    {
        int spls = _Ws.size() / 4;
        return std::accumulate(_Ws.end() - spls, _Ws.end(), 0.0) / (double)(spls);
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
    double _p;
    int _r, _n;
    std::minstd_rand _rnd;
    std::vector<int> _Is, _Ws, _Ts;
    int _T, _sampling;
    std::vector<int> _seq;

    int step()
    {
        if (_n <= 1) return 0;

        int a, b;
        do {
            a = std::uniform_int_distribution<>(0, _n - 1)(_rnd);
            b = std::uniform_int_distribution<>(a - _r, a + _r)(_rnd);
        } while ((b < 0) || (b >= _n) || (b == a));

        if (b < a)
            std::swap(a, b);
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
            .def("converge_on_I", &RandomSort::converge_on_I)
            .def("ET", &RandomSort::ET)
            .def("I", &RandomSort::I)
            .def("W", &RandomSort::W)
            .def("I_stab", &RandomSort::I_stab)
            .def("W_stab", &RandomSort::W_stab)
            .def("__str__", &RandomSort::info_str)
            .def("__repr__", &RandomSort::info_str)
            ;

        class_<std::vector<int>>("intVector")
            .def(vector_indexing_suite<std::vector<int>>())
            ;

}


