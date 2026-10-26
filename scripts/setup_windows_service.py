"""
scripts/setup_windows_service.py
──────────────────────────────────
Register and manage Mayuri as an always-running Windows service / background process.

Supports:
  1. Task Scheduler (Recommended) — Starts on user logon, auto-restarts on crash,
                                    maintains full WASAPI audio & microphone access,
                                    runs silently with no popup window.
                                    Requires Administrator rights (prompts UAC if needed).
  2. Registry RunKey (No Admin)   — Starts when current user logs on.
  3. Startup Folder (No Admin)    — Creates launcher in shell:startup.
  4. NSSM Service Generator        — Configures Non-Sucking Service Manager commands.
  5. Process Controller            — Start, stop, or check status of Mayuri background process.
  6. Uninstaller                   — Cleans up tasks and startup keys.

Usage:
  python scripts/setup_windows_service.py --task       # Task Scheduler (auto-elevates via UAC if needed)
  python scripts/setup_windows_service.py --runkey     # Registry Run Key (NO Admin required)
  python scripts/setup_windows_service.py --startup    # Windows Startup Folder (NO Admin required)
  python scripts/setup_windows_service.py --start      # Start Mayuri in background now
  python scripts/setup_windows_service.py --stop       # Stop Mayuri background process
  python scripts/setup_windows_service.py --status     # Check current status
  python scripts/setup_windows_service.py --uninstall  # Remove from startup/scheduler
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import winreg
from pathlib import Path

# Fix Windows console UTF-8 output if cp1252 is used
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent


def is_admin() -> bool:
    """Check if the current process has Administrator privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def find_python_executable(gui: bool = False) -> str:
    """Detect the best Python executable (virtualenv or current interpreter)."""
    exe_name = "pythonw.exe" if gui else "python.exe"

    # 1. Check local .venv
    local_venv = ROOT / ".venv" / "Scripts" / exe_name
    if local_venv.exists():
        return str(local_venv)

    # 2. Check current sys.executable
    curr_exe = Path(sys.executable)
    if gui:
        curr_gui = curr_exe.parent / "pythonw.exe"
        if curr_gui.exists():
            return str(curr_gui)
    if curr_exe.exists():
        return str(curr_exe)

    # 3. Check known CUDA_ENV parent
    cuda_exe = Path("D:/CUDA_ENV/CUDA_ENV/Scripts") / exe_name
    if cuda_exe.exists():
        return str(cuda_exe)

    return exe_name


def register_task_scheduler(elevated: bool = False) -> None:
    """
    Create a Windows Task Scheduler task that runs Mayuri on logon.
    Uses wscript.exe + run_background.vbs to launch silently in the background.
    """
    task_name = "MayuriAI_Service"
    vbs_path = ROOT / "scripts" / "run_background.vbs"

    if not vbs_path.exists():
        print(f"✗ Error: {vbs_path} does not exist.")
        return

    # Check if elevated
    if not is_admin() and not elevated:
        print("Task Scheduler requires Administrator privileges to register logon tasks.")
        print("Attempting to prompt Windows User Account Control (UAC) for elevation...")
        try:
            # Trigger UAC elevation modal
            ret = ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable,
                f'"{Path(__file__).resolve()}" --task --elevated',
                str(ROOT), 1
            )
            if ret > 32:
                print("✓ Elevation prompt dispatched. Please approve the UAC modal.")
                return
        except Exception as exc:
            print(f"Could not request elevation: {exc}")

        print("\nIf the UAC prompt did not appear:")
        print("1. Open PowerShell as Administrator (Win + X -> Terminal/PowerShell Admin)")
        print(f"   cd \"{ROOT}\"")
        print(f"   python scripts/setup_windows_service.py --task")
        print("\nOR use the NO-ADMIN alternative (already supported):")
        print(f"   python scripts/setup_windows_service.py --runkey")
        return

    # Using schtasks to create an ONLOGON task
    cmd = [
        "schtasks", "/create",
        "/tn", task_name,
        "/tr", f'wscript.exe "{vbs_path}"',
        "/sc", "ONLOGON",
        "/rl", "HIGHEST",
        "/f",  # force overwrite if exists
    ]

    print(f"Creating Task Scheduler entry '{task_name}'...")
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode == 0:
        print(f"✓ Task Scheduler task '{task_name}' successfully created!")
        print("  • Trigger: On Windows User Logon")
        print("  • Mode: Silent background process (no console window)")
        print("  • Audio/Mic: Full WASAPI microphone access enabled in active user session")
        print("\nTo start it immediately, run:")
        print(f"  schtasks /run /tn {task_name}")
    else:
        print(f"✗ Task Scheduler registration failed:")
        print(f"  {res.stderr.strip() or res.stdout.strip()}")
        print("\nAlternative (No Administrator required):")
        print("  python scripts/setup_windows_service.py --runkey")


def register_run_key() -> None:
    """Register Mayuri in HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run."""
    vbs_path = ROOT / "scripts" / "run_background.vbs"
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    value_name = "MayuriAI"
    value_data = f'wscript.exe "{vbs_path}"'

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, value_data)

    print(f"✓ Registered Windows Run Key: {value_name} → {value_data}")
    print("  Mayuri will now start silently in the background whenever you log into Windows.")
    print("  (No Administrator rights required!)")


def register_startup_folder() -> None:
    """Place a silent launcher shortcut/script into the user's Startup folder."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        print("✗ Could not resolve %APPDATA% directory.")
        return

    startup_dir = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    if not startup_dir.exists():
        startup_dir.mkdir(parents=True, exist_ok=True)

    vbs_target = startup_dir / "Mayuri_Startup.vbs"
    source_vbs = ROOT / "scripts" / "run_background.vbs"

    try:
        vbs_target.write_text(source_vbs.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"✓ Created startup launcher: {vbs_target}")
        print("  Mayuri will now start silently on login via Windows Startup folder.")
    except Exception as exc:
        print(f"✗ Failed to create startup file: {exc}")


def print_nssm_guide() -> None:
    """Output exact commands for setting up Mayuri using NSSM (Non-Sucking Service Manager)."""
    python_exe = find_python_executable(gui=False)
    print("\n═════════════════════════════════════════════════════════════════════")
    print("  NSSM (Non-Sucking Service Manager) Setup Guide for Mayuri")
    print("═════════════════════════════════════════════════════════════════════")
    print("1. Download NSSM from https://nssm.cc/download and place nssm.exe in PATH.")
    print("2. Open Administrator PowerShell and run:")
    print(f'   nssm install MayuriAI "{python_exe}" "-m mayuri.main"')
    print(f'   nssm set MayuriAI AppDirectory "{ROOT}"')
    print(f'   nssm set MayuriAI AppStdout "{ROOT / "logs" / "nssm_stdout.log"}"')
    print(f'   nssm set MayuriAI AppStderr "{ROOT / "logs" / "nssm_stderr.log"}"')
    print('   nssm set MayuriAI Start SERVICE_AUTO_START')
    print('   nssm set MayuriAI AppRestartDelay 5000')
    print("\n⚠️ IMPORTANT NOTE ON AUDIO & SESSION 0 ISOLATION:")
    print("   Standard Windows Services run under Session 0 without audio device access.")
    print("   If you configure NSSM, configure it to run under your active Windows user:")
    print('   nssm set MayuriAI ObjectName "YOUR_USERNAME" "YOUR_PASSWORD"')
    print("\n3. To start the service:")
    print("   nssm start MayuriAI")
    print("═════════════════════════════════════════════════════════════════════\n")


def start_service() -> None:
    """Launch Mayuri silently in the background immediately."""
    vbs_path = ROOT / "scripts" / "run_background.vbs"
    if not vbs_path.exists():
        print(f"✗ Error: {vbs_path} does not exist.")
        return

    print("🚀 Launching Mayuri silently in the background...")
    subprocess.Popen(["wscript.exe", str(vbs_path)], cwd=str(ROOT))
    print("✓ Launched. Checking process status in 2 seconds...")
    import time
    time.sleep(2)
    check_status()


def stop_service() -> None:
    """Cleanly stop running Mayuri processes."""
    print("🛑 Stopping Mayuri background processes...")
    try:
        subprocess.run(
            ['powershell', '-Command', "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*mayuri.main*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
            capture_output=True, text=True
        )
        print("✓ All background Mayuri processes stopped.")
    except Exception as exc:
        print(f"Error stopping processes: {exc}")


def check_status() -> None:
    """Inspect whether Mayuri is currently running."""
    print("🔍 Checking Mayuri status...")

    # Check Task Scheduler
    ts_check = subprocess.run(
        ["schtasks", "/query", "/tn", "MayuriAI_Service", "/fo", "LIST"],
        capture_output=True, text=True
    )
    if ts_check.returncode == 0:
        print("✓ Task Scheduler task 'MayuriAI_Service' is REGISTERED.")
    else:
        print("○ Task Scheduler task 'MayuriAI_Service' is not registered.")

    # Check Run Key
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ) as key:
            val, _ = winreg.QueryValueEx(key, "MayuriAI")
            print(f"✓ Windows Run Key 'MayuriAI' is REGISTERED: {val}")
    except FileNotFoundError:
        print("○ Windows Run Key 'MayuriAI' is not registered.")

    # Check running processes
    proc_check = subprocess.run(
        ['powershell', '-Command', "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*mayuri.main*' } | Select-Object ProcessId, CommandLine"],
        capture_output=True, text=True
    )
    if proc_check.returncode == 0 and "mayuri.main" in proc_check.stdout:
        print("\n🟢 Mayuri is CURRENTLY RUNNING:")
        print(proc_check.stdout.strip())
    else:
        print("\n⚪ Mayuri is not currently running as an active process.")


def unregister() -> None:
    """Remove Mayuri from Windows Startup and Task Scheduler."""
    # 1. Run key
    try:
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, "MayuriAI")
        print("✓ Removed Windows Run key.")
    except FileNotFoundError:
        pass

    # 2. Startup folder
    appdata = os.environ.get("APPDATA")
    if appdata:
        target = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "Mayuri_Startup.vbs"
        if target.exists():
            target.unlink()
            print("✓ Removed Startup folder launcher.")

    # 3. Task scheduler
    for name in ("MayuriAI_Service", "MayuriAI"):
        subprocess.run(["schtasks", "/delete", "/tn", name, "/f"], capture_output=True)
    print("✓ Removed Task Scheduler tasks.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mayuri Windows Service Manager")
    parser.add_argument("--task", action="store_true", help="Register as Windows Task Scheduler task (Requires Admin)")
    parser.add_argument("--elevated", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runkey", action="store_true", help="Register in HKCU Run Key (NO Admin required)")
    parser.add_argument("--startup", action="store_true", help="Register in Windows Startup folder (NO Admin required)")
    parser.add_argument("--start", action="store_true", help="Start Mayuri in background immediately")
    parser.add_argument("--stop", action="store_true", help="Stop Mayuri background processes")
    parser.add_argument("--nssm", action="store_true", help="Print NSSM Windows Service installation instructions")
    parser.add_argument("--status", action="store_true", help="Check status of Mayuri service and running process")
    parser.add_argument("--uninstall", action="store_true", help="Remove all startup and scheduler registrations")

    args = parser.parse_args()

    if args.uninstall:
        unregister()
    elif args.task:
        register_task_scheduler(elevated=args.elevated)
    elif args.runkey:
        register_run_key()
    elif args.startup:
        register_startup_folder()
    elif args.start:
        start_service()
    elif args.stop:
        stop_service()
    elif args.nssm:
        print_nssm_guide()
    elif args.status:
        check_status()
    else:
        print("Mayuri Windows Service Setup")
        print("----------------------------")
        print("1. Task Scheduler (Admin):     python scripts/setup_windows_service.py --task")
        print("2. Run Key (No Admin needed):  python scripts/setup_windows_service.py --runkey")
        print("3. Startup Folder (No Admin):  python scripts/setup_windows_service.py --startup")
        print("4. Start in Background Now:    python scripts/setup_windows_service.py --start")
        print("5. Stop Background Process:    python scripts/setup_windows_service.py --stop")
        print("6. Check Status:               python scripts/setup_windows_service.py --status")
        print("7. NSSM Service Guide:         python scripts/setup_windows_service.py --nssm")
        print("8. Uninstall All:              python scripts/setup_windows_service.py --uninstall")


if __name__ == "__main__":
    main()
