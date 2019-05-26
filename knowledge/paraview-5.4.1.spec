%include inc/require-not_system_compiler
%include inc/require-mpi

NAME:           ParaView-5_4_1
VERSION:        5.4.1
RELEASE:        2
SUMMARY:        ParaView Scientific Visualization
LICENSE:        ParaView License v1.2. BSD 3-Clause license.  Licenses of other applications levereged by Paraview or developed by collaborators apply.
GROUP:          Research/Development
URL:            http://paraview.org/

%include inc/setup

SOURCE0:        %{cname}-v%{version}.tar.gz

%description
%include inc/description

ParaView is an open-source, multi-platform data analysis and visualization application. ParaView users can quickly build visualizations to analyze their data using qualitative and quantitative techniques. The data exploration can be done interactively in 3D or programmatically using batch processing capabilities. ParaView was developed to analyze extremely large datasets using distributed memory computing resources.

The 'paraview' executable is the client in the client/server model and typically should not be run on a cluster.  Instead, run 'pvserver' on the cluster and connect via a ParaView client on your local computer. Detailed instructions for our systems are available on the ARC website.

%prep
%include inc/prep

%setup -n %{cname}-v%{version}


%build
%include inc/build

#Currently Loaded Modules:
#  1) gcc/5.2.0   2) openmpi/2.0.0   3) qt/4.8.4   4) cmake/3.8.2   5) python/2.7.10
module add gcc/5.2.0 cmake/3.8.2 python/2.7.10 qt/4.8.4 openmpi/2.0.0 hdf5/1.8.15
export myMPILIB=/opt/apps/gcc5_2/openmpi/2.0.0/lib
export myPYTHON_INC=/opt/apps/gcc5_2/python/2.7.10/include/python2.7
export myPYTHON_DIR=/opt/apps/gcc5_2/python/2.7.10
export myPYTHON_LIB=/opt/apps/gcc5_2/python/2.7.10/lib
export myPYTHON_BIN=/opt/apps/gcc5_2/python/2.7.10/bin
export myHDF5_DIR=/opt/apps/gcc5_2/hdf5/1.8.15

mkdir -p build
cd build
#%{INSTALL_DIR}=/opt/apps/gcc5_2/openmpi1_8/ParaView/5.4.1
cmake -DVTK_RENDERING_BACKEND=OpenGL2 \
      -DBUILD_TESTING=OFF \
      -DCMAKE_INSTALL_PREFIX=%{INSTALL_DIR} \
      -DPARAVIEW_ENABLE_PYTHON=ON \
      -DPARAVIEW_USE_MPI=ON \
      -DMPI-C-LIBRARIES=$myMPILIB \
      -DMPI_CXX_LIBRARIES=$myMPILIB \
      -DMPI_Fortran_LIBRARIES=$myMPILIB \
      -DMPI-LIBRARY=$myMPILIB \
      -DPARAVIEW_ENABLE_PYTHON=ON \
      -DPYTHON_EXECUTABLE=$myPYTHON_BIN/python \
      -DPYTHON_INCLUDE_DIR=$myPYTHON_INC \
      -DPYTHON_LIBRARY=$myPYTHON_LIB/libpython2.7.so \
      -DVTK_USE_SYSTEM_HDF5=ON \
      -DHDF5_DIR=myHDF5_DIR \
      -DCMAKE_BUILD_TYPE=Release ..
make -j8
make DESTDIR="$RPM_BUILD_ROOT" install
cd ..


# append to module file
cat >> "%{MODULE_FILE}" <<'EOF'
help([[
Define Environment Variables:

          $PARAVIEW_DIR - root
          $PARAVIEW_BIN - binaries
          $PARAVIEW_LIB - library
          $LIBGL_ALWAYS_INDIRECT

Prepend Environment Variables:

               PATH += %{INSTALL_DIR}/bin
            INCLUDE += %{INSTALL_DIR}/include
    LD_LIBRARY_PATH += %{INSTALL_DIR}/lib
            MANPATH += %{INSTALL_DIR}/share/man
]])

prepend_path("PATH", "%{INSTALL_DIR}/bin")
prepend_path("INCLUDE", "%{INSTALL_DIR}/include")
prepend_path("LD_LIBRARY_PATH", "%{INSTALL_DIR}/lib")
prepend_path("MANPATH", "%{INSTALL_DIR}/share/man")
prereq("qt","python/2.7.10")

setenv("LIBGL_ALWAYS_INDIRECT", "y")
setenv("PARAVIEW_DIR", "%{INSTALL_DIR}")
setenv("PARAVIEW_BIN", "%{INSTALL_DIR}/bin")
setenv("PARAVIEW_LIB", "%{INSTALL_DIR}/lib")

if os.getenv("DISPLAY") == nil then
    setenv("DISPLAY", ":0")
end
EOF

##chown root:arcadm %{MODULE_DIR}
##chmod -R o-rwx %{INSTALL_DIR} %{MODULE_DIR}
##chown root:ansys /opt/apps/ansys
##chmod o-rwx /opt/apps/ansys


%files -n %{package_name}
%include inc/files


%post -n %{package_name}
%include inc/post


%postun -n %{package_name}
%include inc/postun


%clean
%include inc/clean
