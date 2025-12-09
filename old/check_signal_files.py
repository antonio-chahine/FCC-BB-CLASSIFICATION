#!/usr/bin/env python3
import glob
import os

SIGNAL_GLOB = "/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_wz3p6_ee_qq_ecm91p2/*.root"

def main():
    print("Checking signal files...")
    files = glob.glob(SIGNAL_GLOB)

    print(f"\n🔹 Total files found: {len(files)}\n")

    # Check readability with os.stat() (fast & safe)
    readable = []
    unreadable = []

    for f in files:
        try:
            os.stat(f)
            readable.append(f)
        except Exception:
            unreadable.append(f)

    print(f"🟢 Readable:   {len(readable)}")
    print(f"🔴 Unreadable: {len(unreadable)}\n")

    if unreadable:
        print("Unreadable files:")
        for f in unreadable:
            print("   ", f)

    print("\nList of ALL signal files:")
    for f in readable:
        print("   ", f)

if __name__ == "__main__":
    main()
