from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension('mqConnections', ['/app/mqConnections.py'], \
        extra_compile_args=['-O2', '-g0']),
    Extension('observerFile', ['/app/observerFile.py'], \
        extra_compile_args=['-O2', '-g0']),
    Extension('code_deploy_v6', ['/app/code_deploy_v6.py'], \
        extra_compile_args=['-O2', '-g0']),
    Extension('ctype_utils', ['/app/ctype_utils.py'], \
        extra_compile_args=['-O2', '-g0']),
    Extension('process_utils', ['/app/process_utils.py'], \
        extra_compile_args=['-O2', '-g0']),
    Extension('classifierv2_module', ['/app/classifierv2_module.py'], \
        extra_compile_args=['-O2', '-g0']),
    Extension('configReader', ['/app/configReader.py'], \
        extra_compile_args=['-O2', '-g0']),
    Extension('extended_apilayer', [
              '/app/extended_apilayer.py'], extra_compile_args=['-O2', '-g0'])
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"} #all are Python-3
setup(
    name='classifierv2_module', \
    ext_modules=ext_modules, \
    cmdclass={'build_ext': build_ext}
)