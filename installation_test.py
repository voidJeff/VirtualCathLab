import PyQt6.QtCore as qc
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

print("PyQt6 version:", qc.PYQT_VERSION_STR)
print("Qt version   :", qc.QT_VERSION_STR)
print("VTK version  :", vtk.vtkVersion.GetVTKVersion())