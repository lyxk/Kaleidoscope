# clang++ -mlinker-version=450.3 -g `llvm-config --cxxflags --ldflags --system-libs --libs core` $1.cpp /usr/local/Cellar/llvm/10.0.0_3/lib/*.a -o $1
clang++ -mlinker-version=450.3 -g `llvm-config --cxxflags --ldflags --system-libs --libs` $1.cpp -o $1
