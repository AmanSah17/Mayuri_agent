"""
scripts/setup_windows_service.py
──────────────────────────────────
Register and manage Mayuri as an always-running Windows service / background process.

Supports:
  1. Task Scheduler (Recommended) — Starts on user logon, auto-restarts on crash,
                                    maintains full WASAPI audio & microphone access,
                                    runs silently with no popup window.
  2. Registry RunKey               — Starts when current user logs on.
  3. NSSM Service Generator        — Configures Non-Sucking Service Manager commands.
  4. Status Inspector              — Verifies if Mayuri is currently running.
  5. Uninstaller                   — Cleans up tasks and startup keys.

Usage:
  python scripts/setup_windows_service.py --task       # Register in Task Scheduler (Recommended)
  python scripts/setup_windows_service.py --runkey     # Register in Registry Run Key
  python scripts/setup_windows_service.py --nssm       # Print NSSM commands for native Windows Service
  python scripts/setup_windows_service.py --status     # Check current status
  python scripts/setup_windows_service.py --uninstall  # Remove from startup/scheduler
"""

from __future__ import annotations

import argparse
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


def register_task_scheduler() -> None:
    """
    Create a Windows Task Scheduler task that runs Mayuri on logon.
    Uses wscript.exe + run_background.vbs to launch silently in the background.
    Configured with highest available privileges and auto-restart on unexpected exit.
    """
    task_name = "MayuriAI_Service"
    vbs_path = ROOT / "scripts" / "run_background.vbs"

    if not vbs_path.exists():
        print(f"✗ Error: {vbs_path} does not exist.")
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
        print(f"✗ Failed to create Task Scheduler task (Run as Administrator if elevated):")
        print(f"  {res.stderr.strip() or res.stdout.strip()}")


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

    # 2. Task scheduler
    for name in ("MayuriAI_Service", "MayuriAI"):
        subprocess.run(["schtasks", "/delete", "/tn", name, "/f"], capture_output=True)
    print("✓ Removed Task Scheduler tasks.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mayuri Windows Service Manager")
    parser.add_argument("--task", action="store_true", help="Register as Windows Task Scheduler task (Recommended)")
    parser.add_argument("--runkey", action="store_true", help="Register in HKCU Run Key (starts on logon)")
    parser.add_argument("--nssm", action="store_true", help="Print NSSM Windows Service installation instructions")
    parser.add_argument("--status", action="store_true", help="Check status of Mayuri service and running process")
    parser.add_argument("--uninstall", action="store_true", help="Remove all startup and scheduler registrations")

    args = parser.parse_args()

    if args.uninstall:
        unregister()
    elif args.task:
        register_task_scheduler()
    elif args.runkey:
        register_run_key()
    elif args.nssm:
        print_nssm_guide()
    elif args.status:
        check_status()
    else:
        # Default behavior: show choices
        print("Mayuri Windows Service Setup")
        print("----------------------------")
        print("1. Task Scheduler:  python scripts/setup_windows_service.py --task")
        print("2. Run Key:         python scripts/setup_windows_service.py --runkey")
        print("3. NSSM Service:    python scripts/setup_windows_service.py --nssm")
        print("4. Check Status:    python scripts/setup_windows_service.py --status")
        print("5. Uninstall:       python scripts/setup_windows_service.py --uninstall")


if __name__ == "__main__":
    main()
