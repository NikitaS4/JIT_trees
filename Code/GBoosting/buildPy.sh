# compile project with pybind to .so target

# old compile (works on Linux only)
#c++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` common/*.cpp pybind/*.cpp pybind/*.h common/*.h -o JITtrees`python3-config --extension-suffix`

# new compile (use python script)
# ATTENTION: this way build is running in quiet mode
# so warnings will be invisible
# To see compilation warnings (-Wall), remove '-q' option or
# uncomment & run the old version
python buildPy.py -q build_ext -i

# uncomment the line below to delete build directory each time
# (not recommended in case You run this script frequently)
#rm -r build

# report
echo build finished

