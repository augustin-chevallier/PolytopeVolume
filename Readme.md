## Dependencies

CGAL, Mpfr, Mpfi, Eigen, boost


## Build

Use the commands
```
mkdir build
cd build
cmake ..
make
```

## Usage
To get a list of options:
```
./VolumeComputation.exe --help
```

By default, VolumeComputation.exe will compute the volume of the polytope associated to the matrix A and vector b contained in the files A.txt and b.txt. An example of such files is available in the example folder.

