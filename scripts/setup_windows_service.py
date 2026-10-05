"""
scripts/setup_windows_service.py
──────────────────────────────────
Register Mayuri as a Windows startup item using the Windows Registry.

Two options:
  1. Registry RunKey   — starts with user login (simplest)
  2. Task Scheduler    — can start before login, restart on crash

Run as Administrator for Task Scheduler option.
"""

from __future__ import annotations

import subprocess
import sys
import winreg
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
MAYURI_SCRIPT = ROOT / ".venv" / "Scripts" / "mayuri.exe"


def register_run_key() -> None:
    """Add Mayuri to HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run."""
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    value_name = "MayuriAI"
    value_data = f'"{MAYURI_SCRIPT}"'

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE
    ) as key:
        winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, value_data)

    print(f"✓ Registered Run key: {value_name} → {value_data}")


def register_task_scheduler() -> None:
    """Create a Task Scheduler task that starts Mayuri at login."""
    task_name = "MayuriAI"
    cmd = [
        "schtasks", "/create", "/tn", task_name,
        "/tr", str(MAYURI_SCRIPT),
        "/sc", "ONLOGON",
        "/rl", "LIMITED",
        "/f",   # overwrite if exists
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Task Scheduler task '{task_name}' created.")
    else:
        print(f"✗ Task Scheduler failed: {result.stderr}")


def unregister() -> None:
    """Remove Mayuri from startup."""
    # Run key
    try:
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, "MayuriAI")
        print("✓ Removed Run key.")
    except FileNotFoundError:
        pass

    # Task scheduler
    subprocess.run(["schtasks", "/delete", "/tn", "MayuriAI", "/f"], capture_output=True)
    print("✓ Removed Task Scheduler task (if it existed).")


if __name__ == "__main__":
    if "--uninstall" in sys.argv:
        unregister()
    elif "--task" in sys.argv:
        register_task_scheduler()
    else:
        register_run_key()
        print("\nUse --task for Task Scheduler (requires Admin)")
        print("Use --uninstall to remove startup entry")
