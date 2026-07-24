# Aleph-0: Visualizing Countable Infinity

An interactive 3D visualization of countable infinity (ℵ₀) built with Three.js. Explore the beauty of mathematical infinity through six interconnected visualizations.

**Live demo:** [Open in browser](index.html) — no build step required!

## Visualizations

### 1. Number Spiral
Natural numbers 1, 2, 3, ... arranged on a 3D helix, appearing one by one. The spiral never ends — just like the natural numbers it represents.

### 2. Hilbert's Hotel
The famous paradox of countable infinity: a fully occupied infinite hotel where a new guest can always be accommodated by shifting every guest one room over.

### 3. One-to-One Correspondence
Shows bijections between ℕ (naturals), 2ℕ (evens), and P (primes) — proving all three infinite sets have the same cardinality ℵ₀.

### 4. Menger Sponge
A fractal with infinite surface area but zero volume. Demonstrates how infinity can be structural, with infinitely many recursive holes.

### 5. Cantor's Diagonal Argument
Proves the real numbers are *uncountable* by constructing a binary string that differs from every row in an infinite table — showing |ℝ| > ℵ₀.

### 6. Rational Number Enumeration
Demonstrates that ℚ (rationals) are countable by placing fractions p/q on a 2D grid and enumerating along diagonals.

## Controls

| Action | Input |
|--------|-------|
| Rotate | Left-drag |
| Pan | Right-drag |
| Zoom | Scroll wheel |
| Switch modes | Keys `1`–`6` |
| Play/Pause | `Space` |
| Fullscreen | `F` |
| Reset | `R` |

## Features

- **6 interactive modes** covering major concepts of countable infinity
- **Smooth camera transitions** between visualization modes
- **Responsive design** — works on desktop, tablet, and mobile
- **Accessibility** — ARIA labels, keyboard navigation, screen reader support
- **Speed control** — Adjust animation speed with +/- buttons
- **Touch support** — Pinch-to-zoom and touch rotation
- **No build step** — Pure HTML/JS, open `index.html` in any modern browser

## Architecture

```
aleph-0/
├── index.html         # Main HTML with CSS variables, responsive layout, accessibility
├── app.js             # Three.js application (6 visualization modes, orbit controls, UI)
├── three.min.js       # Three.js library (r128)
├── OrbitControls.js   # (unused — custom orbit controls implemented in app.js)
└── README.md          # This file
```

## Mathematical Concepts

This project demonstrates several fundamental concepts from set theory and real analysis:

- **Aleph-0 (ℵ₀)** — The cardinality of the countable infinite sets
- **Cardinality** — The "size" of sets, measured by bijections
- **Countability** — A set is countable if its elements can be listed in a sequence
- **Bijection** — A one-to-one, onto mapping between sets
- **Cantor's Diagonal Argument** — Proof that the reals are uncountable
- **Menger Sponge** — A fractal with infinite surface area and zero volume

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 14+
- Edge 80+

Requires WebGL support. Uses Three.js (r128) from bundled file.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Credits

- Three.js by [mrdoob](https://github.com/mrdoob)
- Math concepts inspired by David Hilbert, Georg Cantor, and Karl Menger