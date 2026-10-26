"""Mount Google Drive on the Colab VM with a long wait.

The CLI's own drivemount command leaves drive.mount() at its 120-second
default, which is too short for a browser OAuth grant. This asks for 15
minutes instead.
"""
from google.colab import drive

drive.mount("/content/drive", timeout_ms=900000, force_remount=True)
print("MOUNT_DONE", flush=True)
import os
print("MYDRIVE:", sorted(os.listdir("/content/drive/MyDrive"))[:40], flush=True)
