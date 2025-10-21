#!/usr/bin/env python3
import matplotlib.pyplot as plt
import math, random
from matplotlib.collections import LineCollection

def load_lsystem(filename):
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    axiom, angle, init_dir = lines[0].split()
    rules = {}
    for line in lines[1:]:
        if "->" in line:
            left, right = line.split("->")
            rules[left.strip()] = right.strip()
    return axiom, float(angle), float(init_dir), rules

def generate(axiom, rules, iterations):
    s = axiom
    for _ in range(iterations):
        s = "".join(rules.get(ch, ch) for ch in s)
    return s

def draw_lsystem(ax, sequence, angle, init_dir, base_length=12):
    stack = []
    x, y = 0.0, 0.0
    current_angle = init_dir
    lines = []
    colors = []
    widths = []

    for ch in sequence:
        if ch in "Ff":
            depth = len(stack)  # используем длину стека для цвета и толщины
            length = base_length * (0.75 ** depth)
            thickness = max(0.5, 6 * (0.8 ** depth))
            x2 = x + math.cos(math.radians(current_angle)) * length
            y2 = y + math.sin(math.radians(current_angle)) * length

            if ch == "F":
                t = min(1, depth / 10)  # делаем градиент заметным
                r = 0.4 * (1 - t) + 0.2 * t
                g = 0.2 * (1 - t) + 0.8 * t
                b = 0.05 * (1 - t) + 0.2 * t
                lines.append([(x, y), (x2, y2)])
                colors.append((r, g, b))
                widths.append(thickness)

            x, y = x2, y2

        elif ch == "+":
            current_angle += angle + random.uniform(-10, 10)
        elif ch == "-":
            current_angle -= angle + random.uniform(-10, 10)
        elif ch == "[":
            stack.append((x, y, current_angle))
        elif ch == "]" and stack:
            x, y, current_angle = stack.pop()

    lc = LineCollection(lines, colors=colors, linewidths=widths, capstyle='round')
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    return ax

def main():
    filename = r"d:\ss\CompGraf\Lab5\Dodukalov_Task1\tree.txt"
    axiom, angle, init_dir, rules = load_lsystem(filename)
    seq = generate(axiom, rules, iterations=6)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")
    draw_lsystem(ax, seq, angle, init_dir)
    plt.title(f"Фрактальное дерево tree.txt")
    plt.show()

if __name__ == "__main__":
    main()
