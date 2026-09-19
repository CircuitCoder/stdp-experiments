{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [
    (python313.withPackages (ps: with ps; [ numpy scipy pytest psutil pybind11 setuptools cython ]))
    gcc gnumake cmake nest gsl libtool readline git util-linux procps pkg-config
  ];
  shellHook = ''
    export PYTHONPATH="$PWD/3rdparty/genn:$PWD/reimpl:$PWD/brunel:$PWD/genn-sweep:$PWD/cpu-benchmark:$PYTHONPATH"
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.libffi pkgs.zlib ]}:$LD_LIBRARY_PATH"
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export CXX="g++ -O3 -march=native"
  '';
}
