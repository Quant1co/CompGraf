#!/usr/bin/env python3
import math, random, argparse
import matplotlib.pyplot as plt

# --- чтение описания из файла ---
def read_description(path):
    lines = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip() and not ln.strip().startswith("#")]
    header = lines[0].split()
    axiom, angle, direction = header[0], float(header[1]), float(header[2])
    rules = {}
    for ln in lines[1:]:
        if "->" not in ln:
            continue
        left, right = [p.strip() for p in ln.split("->", 1)]
        alts = []
        for alt in right.split("|"):
            alt = alt.strip()
            if ":" in alt:
                rep, w = alt.rsplit(":", 1)
                alts.append((rep, float(w)))
            else:
                alts.append((alt, 1.0))
        rules[left] = alts
    return {"axiom": axiom, "angle": angle, "direction": direction, "rules": rules}

# --- разворачивание L-системы ---
def expand_lsystem(axiom, rules, iterations):
    s = axiom
    for _ in range(iterations):
        new_s = []
        for ch in s:
            if ch in rules:
                alts = rules[ch]
                total = sum(w for _, w in alts)
                r = random.random() * total
                acc = 0
                for rep, w in alts:
                    acc += w
                    if r <= acc:
                        new_s.append(rep)
                        break
            else:
                new_s.append(ch)
        s = "".join(new_s)
    return s

# --- интерпретация L-системы как "черепаха" ---
def interpret_turtle(s, angle, step=1.0, init_dir=0.0):
    x, y = 0.0, 0.0
    dir_rad = math.radians(init_dir)
    stack = []
    segs = []
    for ch in s:
        if ch in "FG":
            nx = x + math.cos(dir_rad) * step
            ny = y + math.sin(dir_rad) * step
            segs.append((x, y, nx, ny))
            x, y = nx, ny
        elif ch == "f":
            x += math.cos(dir_rad) * step
            y += math.sin(dir_rad) * step
        elif ch == "+":
            dir_rad += math.radians(angle)
        elif ch == "-":
            dir_rad -= math.radians(angle)
        elif ch == "[":
            stack.append((x, y, dir_rad))
        elif ch == "]" and stack:
            x, y, dir_rad = stack.pop()
    return segs

# --- отрисовка ---
def plot_segments(segs, title="L-system"):
    xs = [p for s in segs for p in (s[0], s[2])]
    ys = [p for s in segs for p in (s[1], s[3])]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    w, h = maxx - minx, maxy - miny
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    for (x1, y1, x2, y2) in segs:
        ax.plot([x1, x2], [y1, y2], "k", lw=0.8)
    ax.axis("off")
    ax.set_xlim(minx - 0.1 * w, maxx + 0.1 * w)
    ax.set_ylim(miny - 0.1 * h, maxy + 0.1 * h)
    ax.set_title(title)
    plt.show()

# --- main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", required=True)
    parser.add_argument("--iter", "-n", type=int, default=5)
    args = parser.parse_args()

    desc = read_description(args.file)
    s = expand_lsystem(desc["axiom"], desc["rules"], args.iter)
    segs = interpret_turtle(s, desc["angle"], 1.0, desc["direction"])
    plot_segments(segs, f"L-system {args.file}")

if __name__ == "__main__":
    main()
