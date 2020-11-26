# compile project with pybind to .so target

c++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` common/*.cpp pybind/*.cpp pybind/*.h common/*.h -o JITtrees`python3-config --extension-suffix`

# compile project to executable
g++ -Wall -std=c++17 common/*.cpp common/*.h cppMain/*.cpp -o JITtrees

# report
echo build finished

