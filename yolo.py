import argparse
import glob
import os
import struct

def read_obj_vertices(path: str):
    verts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # OBJ vertex position line: v x y z
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return verts

def write_mdd(out_path: str, frames_verts, fps: float):
    """
    MDD (Lightwave Point Cache) is typically big-endian:
      int32 totalFrames
      int32 totalPoints
      float32 times[totalFrames]
      float32 points[totalFrames][totalPoints][3]
    (Big-endian note is widely referenced for MDD.) :contentReference[oaicite:1]{index=1}
    """
    total_frames = len(frames_verts)
    if total_frames == 0:
        raise RuntimeError("No frames to write.")

    total_points = len(frames_verts[0])
    for i, fv in enumerate(frames_verts):
        if len(fv) != total_points:
            raise RuntimeError(f"Vertex count mismatch at frame {i}: {len(fv)} vs {total_points}")

    with open(out_path, "wb") as f:
        # Big-endian ints
        f.write(struct.pack(">ii", total_frames, total_points))

        # times in seconds
        for i in range(total_frames):
            t = float(i) / float(fps)
            f.write(struct.pack(">f", t))

        # vertex positions
        for fv in frames_verts:
            for (x, y, z) in fv:
                f.write(struct.pack(">fff", x, y, z))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder containing OBJ frames")
    ap.add_argument("--pattern", default="*.obj", help="Glob pattern, e.g. mesh_*.obj")
    ap.add_argument("--fps", type=float, default=20.0, help="Frame rate used for time stamps")
    ap.add_argument("--out", required=True, help="Output .mdd path")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    if not paths:
        raise RuntimeError("No OBJ files found (check --in_dir and --pattern).")

    frames = []
    base_n = None
    for p in paths:
        v = read_obj_vertices(p)
        if base_n is None:
            base_n = len(v)
            if base_n == 0:
                raise RuntimeError(f"No vertices read from first OBJ: {p}")
        elif len(v) != base_n:
            raise RuntimeError(f"Vertex count mismatch: {p} has {len(v)} verts, expected {base_n}")
        frames.append(v)

    write_mdd(args.out, frames, args.fps)
    print(f"Wrote MDD: {args.out}")
    print(f"Frames: {len(frames)}, Points: {base_n}, FPS: {args.fps}")

if __name__ == "__main__":
    main()