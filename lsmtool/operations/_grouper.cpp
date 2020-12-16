#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;
typedef double cdt; // coordinate data type, float or double
typedef std::vector<std::pair<cdt,cdt>> ctype; // type for coordinates

class Grouper{

public:
Grouper() : _kernelSize(1.0), _numberOfIterations(10), _lookDistance(1.0), _groupingDistance(1.0){

};
void setKernelSize(double kernelSize){_kernelSize = kernelSize;}
void setNumberOfIterations (int numberOfIterations){_numberOfIterations = numberOfIterations;}
void setLookDistance (double lookDistance){_lookDistance = lookDistance;}
void setGroupingDistance (double groupingDistance){_groupingDistance = groupingDistance;}
void readCoordinates(py::array_t<cdt> array, py::array_t<cdt> farray);
void run();
void group(py::list l);

private:
double gaussian_kernel(double distance);
 ctype _coordinates;
 std::vector<cdt> _fluxes;
 double _kernelSize;
 int _numberOfIterations;
 double _lookDistance;
 double _groupingDistance;

};


void Grouper::readCoordinates(py::array_t<cdt> array, py::array_t<cdt> farray){
  auto arraybuf    = array.request();
  auto farraybuf = farray.request();
  cdt* coordinates = (cdt*) arraybuf.ptr;
  cdt* fluxes = (cdt*) farraybuf.ptr;
  if (arraybuf.ndim != 2 || arraybuf.shape[1] != 2){
    py::print("Grouper::readCoordinates: expected 2D array with two columns!");
  }
  if (farraybuf.ndim != 1){
    py::print("Grouper::readCoordinates: expected 1D array!");
  }
  unsigned N = arraybuf.shape[0];
  if (farraybuf.shape[0] != N){
    py::print("Grouper::readCoordinates: array of fluxes does not have the same length as array of coordinates!");
  }
  for (unsigned n=0; n<N; ++n){
    _coordinates.push_back(std::pair<cdt, cdt>(coordinates[2*n],coordinates[2*n+1]));
    _fluxes.push_back(fluxes[n]);
  }

}

static double euclid_distance(const std::pair<cdt, cdt> &coordinate1, const std::pair<cdt,cdt> &coordinate2){
// Euclidian distance between two points
  double distx = coordinate1.first - coordinate2.first;
  double disty = coordinate1.second - coordinate2.second;
  return sqrt(distx * distx + disty * disty);
}

static std::vector<unsigned> neighbourhood_points(const std::pair<cdt,cdt>& centroid, const ctype& coordinates, double max_distance){
std::vector<unsigned> result;
  for (unsigned n=0; n<coordinates.size(); ++n){
    double distance = euclid_distance(centroid, coordinates[n]);
    if (distance < max_distance){
      result.push_back(n);
    }
  }
return result;
}

double Grouper::gaussian_kernel(double distance){
// Gaussian kernel
  double result = 1.0 / (_kernelSize * std::sqrt(2.0*3.14159265359));
  result *= std::exp(-0.5 * distance * distance / (_kernelSize * _kernelSize));
  return result;
}

void Grouper::run(){
  // run the algorithm

  if (_coordinates.size() == 0){
    py::print("Grouper::run: coordinates have not been read!");
  }


  ctype newcoords = _coordinates;

  for (int it = 0; it < _numberOfIterations; ++it){
#pragma omp parallel for
    for (unsigned n=0; n<_coordinates.size(); ++n){
      std::vector<unsigned> idx_neighbours = neighbourhood_points(_coordinates[n], _coordinates, _lookDistance);
      double denominator = 0.0;
      double numx = 0.0;
      double numy = 0.0;
      for (auto i : idx_neighbours){
         double distance = euclid_distance(_coordinates[i], _coordinates[n]);
         double weight = gaussian_kernel(distance);
         weight *= _fluxes[i];
         numx += weight * _coordinates[i].first;
         numy += weight * _coordinates[i].second;
         denominator +=  weight;
      }
       newcoords[n] = std::pair<cdt,cdt>(numx/denominator, numy/denominator);
    }
    double maxdiff = 0.0;
    unsigned n=0;
    for (auto &coordinate : _coordinates){
      double diff = euclid_distance(coordinate, newcoords[n]);
      if (diff > maxdiff){
        maxdiff = diff;
      }
      ++n;
    }
    _coordinates = newcoords;
    if ((it > 1) && (maxdiff < _groupingDistance / 2.0)){
      break;
    }

  }


}

void Grouper::group(py::list l){

  ctype coords_to_check = _coordinates;

  while (coords_to_check.size() > 0){
    std::vector<unsigned> idx_cluster;
    std::vector<unsigned> idx_cluster_to_remove;
#pragma omp parallel
{
    idx_cluster = neighbourhood_points(coords_to_check[0], _coordinates, _groupingDistance);
    idx_cluster_to_remove = neighbourhood_points(coords_to_check[0], coords_to_check, _groupingDistance);
}
    unsigned n = 0;
    for (auto idx : idx_cluster_to_remove){
      coords_to_check.erase(coords_to_check.begin() + idx - n);
      ++n;
    }

    py::list newcluster;
    for (auto idx : idx_cluster){
      newcluster.append(py::int_(idx));
    }
    l.append(newcluster);

  }
}

PYBIND11_MODULE(_grouper, m)
{
  py::class_<Grouper>(m, "Grouper")
    .def(py::init<>())
    .def("run", &Grouper::run)
    .def("setKernelSize", &Grouper::setKernelSize)
    .def("setNumberOfIterations", &Grouper::setNumberOfIterations)
    .def("setLookDistance", &Grouper::setLookDistance)
    .def("setGroupingDistance", &Grouper::setGroupingDistance)
    .def("readCoordinates", &Grouper::readCoordinates)
    .def("group", &Grouper::group)
;
}


int main(){

  Py_Initialize();

  return 0;
}

