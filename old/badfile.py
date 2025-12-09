#!/usr/bin/env python3
import argparse
import signal
import math
from podio import root_io
import ROOT
ROOT.gROOT.SetBatch(True)

import functions

# ============ Timeout handling ============
class Timeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise Timeout("Operation timed out")

signal.signal(signal.SIGALRM, timeout_handler)

# ============ Utility wrappers ============

def safe_call(name, timeout_sec, func, *args, **kwargs):
    """Run a function with timeout. Returns (ok, result or exception)."""
    signal.alarm(timeout_sec)
    try:
        out = func(*args, **kwargs)
        signal.alarm(0)
        print(f"       {name} OK")
        return True, out
    except Timeout:
        print(f"       {name} HANG/TIMEOUT")
        signal.alarm(0)
        return False, None
    except Exception as e:
        print(f"       {name} ERROR:", e)
        signal.alarm(0)
        return False, e


# ============ Main diagnostic ============

def debug_file(filename):

    print("\n========== DIAGNOSTIC START ==========")
    print("File:", filename)
    print("=======================================\n")
    
    print("Creating ROOT reader...")
    reader = root_io.Reader(filename)
    print("Reader created OK")


    # Load events safely
    ok, events = safe_call("load events", 5, reader.get, "events")
    if not ok:
        print("Could not load event tree; file is corrupt.")
        return

    # Try counting events
    try:
        nevents = len(events)
        print("Total events in file:", nevents)
    except Exception as e:
        print("Error counting events:", e)
        return

    # Loop through events
    for i_event, event in enumerate(events):

        print(f"\n--- Event {i_event} ------------------------")

        # Try loading hit collection
        ok, vb = safe_call("event.get(VertexBarrelCollection)", 5,
                           event.get, "VertexBarrelCollection")
        if not ok:
            print("Skipping event due to hang.")
            continue

        # Try reading size
        try:
            nhits = vb.size()
            print(f"  Hit collection size: {nhits}")
        except Exception as e:
            print("  ERROR reading hit collection size:", e)
            continue

        # Iterate hits
        for i_hit, hit in enumerate(vb):

            print(f"    Hit {i_hit}: start")

            # --- Position ---
            ok, pos = safe_call("getPosition()", 5, hit.getPosition)
            if not ok:
                continue
            print(f"       pos = ({pos.x}, {pos.y}, {pos.z})")

            # --- MC link ---
            ok, mc = safe_call("getMCParticle()", 5, hit.getMCParticle)
            if not ok:
                continue

            if mc is not None:
                # pid
                ok, pid = safe_call("mc.getPDG()", 5, mc.getPDG)
                # energy
                ok2, energy = safe_call("mc.getEnergy()", 5, mc.getEnergy)
            else:
                print("       MC = None")

            # --- getEDep ---
            ok, edep = safe_call("hit.getEDep()", 5, hit.getEDep)

            # --- radius_idx ---
            ok, _ = safe_call("radius_idx()", 5,
                               functions.radius_idx, hit, [14, 36, 58])

            # --- grid index conversion ---
            ok, _ = safe_call("get_grid_indices()", 5,
                               functions.get_grid_indices, pos.x, pos.y, pos.z)

            print("    Hit complete\n")

    print("\n========== DIAGNOSTIC END ==========\n")


# ============ Entry point ============

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True,
                        help="Path to the problematic ROOT file")
    args = parser.parse_args()
    debug_file(args.file)
