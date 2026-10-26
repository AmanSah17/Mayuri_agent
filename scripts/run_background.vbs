' Mayuri Silent Windows Background Launcher
' Starts Mayuri completely headless without any terminal/console window
Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")

' Get project root directory (parent of scripts/)
ScriptDir = FSO.GetParentFolderName(WScript.ScriptFullName)
RootDir = FSO.GetParentFolderName(ScriptDir)

' Detect Python interpreter
PythonExe = RootDir & "\.venv\Scripts\pythonw.exe"
If Not FSO.FileExists(PythonExe) Then
    PythonExe = "D:\CUDA_ENV\CUDA_ENV\Scripts\pythonw.exe"
End If
If Not FSO.FileExists(PythonExe) Then
    PythonExe = "pythonw.exe"
End If

' Run Mayuri module in root working directory, completely hidden (0 = hide window, False = don't wait)
WshShell.CurrentDirectory = RootDir
Command = """" & PythonExe & """ -m mayuri.main"
WshShell.Run Command, 0, False
