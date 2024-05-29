"""The wntr.epanet.toolkit module is a Python extension for the EPANET
Programmers Toolkit DLLs.
"""  # noqa: D205

import ctypes
import logging
import os
import os.path
import platform
import sys
from pkg_resources import resource_filename
from wntr.epanet.toolkit import ENepanet

logger = logging.getLogger(__name__)

epanet_quantum_toolkit = "wntr_quantum.epanet.toolkit"

if os.name in ["nt", "dos"]:
    libepanet = resource_filename(__name__, "Windows/epanet2.dll")
elif sys.platform in ["darwin"]:
    libepanet = resource_filename(__name__, "Darwin/libepanet.dylib")
else:
    libepanet = resource_filename(__name__, "Linux/libepanet2.so")


class ENepanet_quantum(ENepanet):
    """Wrapper class to load the EPANET DLL object, then perform operations on
    the EPANET object that is created when a file is loaded.
    This simulator is thread safe **only** for EPANET `version=2.2`.
    """  # noqa: D205

    def __init__(self, inpfile="", rptfile="", binfile="", version=2.2):
        """Init the class
        Parameters.
            ----------
            inpfile : str
                Input file to use
            rptfile : str
                Output file to report to
            binfile : str
                Results file to generate
            version : float
                EPANET version to use (either 2.0 or 2.2)
        """  # noqa: D205
        super().__init__(inpfile, rptfile, binfile, version)

        if float(version) == 2.0:
            raise NotImplementedError("Not implemented for EPANET2 and only for 2.2")
        elif float(version) == 2.2:
            libnames = ["epanet22", "epanet22_win32"]
            if "64" in platform.machine():
                libnames.insert(0, "epanet22_amd64")
        for lib in libnames:
            try:
                if os.name in ["nt", "dos"]:
                    # libepanet = resource_filename(
                    #     epanet_toolkit, "Windows/%s.dll" % lib
                    # )
                    # self.ENlib = ctypes.windll.LoadLibrary(libepanet)
                    raise NotImplementedError("Not implemented for Windows")

                elif sys.platform in ["darwin"]:
                    # libepanet = resource_filename(
                    #     epanet_toolkit, "Darwin/lib%s.dylib" % lib
                    # )
                    # self.ENlib = ctypes.cdll.LoadLibrary(libepanet)
                    raise NotImplementedError("Not implemented for Darwin")
                else:
                    libepanet = resource_filename(
                        epanet_quantum_toolkit, "Linux/lib%s.so" % lib
                    )
                    print(libepanet)
                    self.ENlib = ctypes.cdll.LoadLibrary(libepanet)
                return
            except Exception as E1:
                if lib == libnames[-1]:
                    raise E1
                pass
            finally:
                if version >= 2.2 and "32" not in lib:
                    self._project = ctypes.c_uint64()
                elif version >= 2.2:
                    self._project = ctypes.c_uint32()
                else:
                    self._project = None
        return
