# Lab 5 Project Report Site

Static engineering report for the UR7e vision-guided pick-and-place project.

## Files

- `index.html` — the report (all sections).
- `styles.css` — typography, layout, and component styles.
- `script.js` — sticky-nav active-section highlight (one IntersectionObserver).
- `assets/` — curated copies of repo images so the site is self-contained.

## How to open

Either:

1. Double-click `index.html` (works in any modern browser, no server needed).
2. Or run a tiny local server from this folder if you want clean URLs:

   ```bash
   python -m http.server 8000
   # then open http://localhost:8000
   ```

## Where to drop final media later

- Replace the dashed "VIDEO / PHOTO / METRICS" cards in **§7 Results &amp; Media**
  with the actual demo video (`<video>` tag or YouTube embed), final hero photo,
  and a small results table.
- The orbit and segmentation galleries currently use frames 1, 10, 15, 20, 25, 30.
  Swap in the best frames from the final run by replacing the matching files in
  `assets/` (keep the same filenames or update `index.html`).
