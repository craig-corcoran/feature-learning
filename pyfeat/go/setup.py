import os.path
import numpy
import distutils.core
import distutils.extension
import Cython.Distutils

def numpy_include_dir():
    path = os.path.dirname(numpy.__file__)

    return os.path.join(path, "core/include") 

distutils.core.setup(
    name = "PyGo",
    ext_modules = [
    distutils.extension.Extension(
            "hash_maps",
            ["hash_maps.pyx"],
            include_dirs = [numpy_include_dir()],
            libraries = ["m", "curses"],
            ),
    distutils.extension.Extension(
            "fuego",
            ["fuego.pyx"],
            language = "c++",
            include_dirs = [
                'fuego/go',
                'fuego/gouct',
                'fuego/smartgame',
                'fuego/simpleplayers',
                numpy_include_dir(),
                ],
            extra_objects = [
                "fuego/go/libfuego_go.a",
                "fuego/gouct/libfuego_gouct.a",
                "fuego/simpleplayers/libfuego_simpleplayers.a",
                "fuego/smartgame/libfuego_smartgame.a",
                ],
            libraries = ["m"],
            ),
        ],
    cmdclass = {'build_ext': Cython.Distutils.build_ext},
    )

