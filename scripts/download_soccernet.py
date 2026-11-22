#!/usr/bin/env python3
"""Download one SoccerNet Game State split (valid). Large. Optional."""
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader

ROOT = Path(__file__).resolve().parents[1]
dest = ROOT / "data" / "SoccerNetGS"
dest.mkdir(parents=True, exist_ok=True)
dl = SoccerNetDownloader(LocalDirectory=str(dest))
print("Downloading SoccerNet Game State (valid). This can take a long time.")
dl.downloadDataTask(task="gamestate-2024", split=["valid"])
print("Done. Unzip under", dest)
