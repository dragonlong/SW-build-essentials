https://wiki.hpc.uconn.edu/index.php/Compiling_software
# computer architecture knowledge
https://www.pgroup.com/support/definitions.htm#x64\
# linking
https://stackoverflow.com/questions/6562403/i-dont-understand-wl-rpath-wl
https://www.gnu.org/software/libtool/manual/html_node/Link-mode.html#Link-mode


# to know better about the problem, you may need to check
# to know which is not working --> could we solve it? --> why it happens --> go through the documentations?
# ask for help!!!
how they work:
cmake VERBOSE=1 -D CMAKE_C_FLAGS="-I$GSL_INCLUDE" -DHAVE_NLOPT=OFF ..
cmake VERBOSE=1 -D CMAKE_C_FLAGS="-I$GSL_INCLUDE" -DHAVE_NLOPT=ON ..
cmake VERBOSE=1 -D CMAKE_C_FLAGS="-I$GSL_INCLUDE" -DHAVE_NLOPT=ON -DNLOPT_LIBRARY=$NLOPT_LIB -LAH ..
cmake [-D<var>=<value>] [-p <cmake-script-file>] src_path
cmake --build <dir>
cmake -E
cmake --find_package <options>
# tricks with cmake
make makefile

cd src && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX="$RPM_BUILD_ROOT/%{INSTALL_DIR}" \
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE \
      -DCMAKE_SKIP_INSTALL_RPATH=ON \
      -DCMAKE_C_FLAGS="-I${GSL_INC} -I${NLOPT_INC}" \
      -DNLOPT_LIBRARY=${NLOPT_LIB}/libnlopt.so    \
      -DHAVE_GSL=ON \
      -DHAVE_NLOPT=ON .

Autoconf flags
./configure $XORG_CONFIG --prefix="$RPM_BUILD_ROOT/%{INSTALL_DIR}/"
$ CFLAGS=-DUSE_DEBUG ./configure [...]
 --disable-shared          Only build static libraries.
 --prefix                  Install all files relative to this directory.
 --enable-gcc-warnings     Enable extra compiler checking with GCC.
 --disable-malloc-replacement
                           Don't let applications replace our memory
                           management functions.
 --disable-openssl         Disable support for OpenSSL encryption.
 --disable-thread-support  Don't support multithreaded environments.

1. use -D in cmake,
gcc/g++ -I  -Wall -W -Wno-unknown-pragmas -O3 -fopenmp -o CMakeFiles/neper.dir/neper.c.o
 -DCMAKE_BUILD_TYPE=Release

export CFLAGS=-ggdb
export CXXFLAGS=-ggdb


2. use set commands/.cmake
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_C_FLAGS "-O0 -ggdb")
set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_C_FLAGS_RELEASE "-O0 -ggdb")
set(CMAKE_CXX_FLAGS "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O0 -ggdb")

3. change CMakeLists.txt

4. change Makefile manually/make include

One more question: Linking Flags
SET(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
SET(GCC_COVERAGE_LINK_FLAGS    "-lgcov")

target_link_options

Please refer to
https://www.selectiveintellect.net/blog/2016/7/29/using-cmake-to-add-third-party-libraries-to-your-project-1
https://pabloariasal.github.io/2018/02/19/its-time-to-do-cmake-right/

Your-external "mylib", add GLOBAL if the imported library is located in directories above the current.
add_library( mylib SHARED IMPORTED )
# You can define two import-locations: one for debug and one for release.
set_target_properties( mylib PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/res/mylib.so )
TARGET_LINK_LIBRARIES(GLBall mylib)
