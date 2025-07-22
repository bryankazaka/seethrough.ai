from pathlib import Path
import humanize
from collections import Counter

root = Path("D:/GenVideo")
real_dir = root / "real"
fake_dir = root / "fake"

# 1) Gather all .mp4 paths
real_paths = list(real_dir.rglob("*.mp4"))
fake_paths = list(fake_dir.rglob("*.mp4"))

# 2) Counts
print(f"✔️ Real clips: {len(real_paths)}")
print(f"✔️ Fake clips: {len(fake_paths)}")

# 3) Disk usage
real_bytes = sum(p.stat().st_size for p in real_paths)
fake_bytes = sum(p.stat().st_size for p in fake_paths)
print(f"📂 Real size: {humanize.naturalsize(real_bytes)}")
print(f"📂 Fake size: {humanize.naturalsize(fake_bytes)}")

# 4) Breakdown by subfolder
def breakdown(paths):
    cnt = Counter(p.parent.name for p in paths)
    for k,v in cnt.most_common():
        print(f"   • {k}: {v}")
print("\n─ Real by source:")
breakdown(real_paths)
print("\n─ Fake by generator:")
breakdown(fake_paths)
