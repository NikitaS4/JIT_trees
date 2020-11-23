#/bin/bash

# compile project
mkdir build
cd build
cmake ..
make

# move executable to the src dir
mv JITtrees ../JITtrees

# delete temporary files
cd ..
rm -r build

