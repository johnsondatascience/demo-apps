# Range Finder — interactive estimator for johnsondatascience.com

A single-file, dependency-free widget for the marketing site. A visitor types in one
number they already know; the page runs a real model on it and hands back a
*distribution* instead of a point guess. The pitch is the method: anyone can put a
growth rate in a spreadsheet, and this makes the difference visible in ten seconds.

`index.html` is the whole thing — no build step, no npm, no framework, no network
calls except the Google Fonts stylesheet.

## What it actually computes

**Tab 1 · Revenue outlook** (Monte Carlo). The anchor number fixes today's level;
three sliders fix drift, volatility and a seasonal factor. 5,000 paths run forward
12 months:

```
e_t       = φ·e_{t-1} + √(1-φ²)·z_t          φ = 0.35, z ~ N(0,1)
base_t    = base_{t-1} · exp(μ + σ·e_t)      μ = ln(1+g)/12
revenue_t = base_t · (1 + (A/2)·cos(2π(m-11)/12))
```

AR(1) innovations mean a bad month raises the odds of another bad month, which is how
real months behave and is the part a spreadsheet always misses. Output is the P10 /
P25 / P50 / P75 / P90 fan, the 12-month total, and P(total > a flat plan).

**Tab 2 · A/B lift test** (Bayesian). Conversion rates get exact
`Beta(1 + conversions, 1 + non-conversions)` posteriors under a uniform prior. The
density curves and the credible intervals are analytic (Lanczos log-gamma, the
regularized incomplete beta by continued fraction, bisection for the quantiles); the
relative lift, P(variant better) and the expected loss come from 20,000 posterior
draws (Marsaglia–Tsang gamma pairs).

This tab covers the same ground as the `bayesian-ab-calculator` repo, deliberately. That one
is Streamlit — it cannot be embedded in a Squarespace page, and its README's live-demo link is
still a placeholder. This is the same model with no server to keep running, so it can sit on
the marketing site permanently.

Both models are seeded from their inputs, so the same numbers always return the same
answer — no jitter when a visitor drags a slider back to where it was.

### Verified against known values

| Check | Result |
|---|---|
| `betaCdf(0.3, 2, 5)` | 0.579825 — matches the analytic value to 6 dp |
| P(variant better), 210/8400 vs 252/8350 | 98.00% vs 97.97% from a normal approximation |
| 12-month total at 1% volatility | 3,192,243 vs 3,191,624 analytic |
| Identical arms | P(better) = 49.5% |
| Same inputs, repeated runs | bit-identical output |

## Putting it on Squarespace

Three ways, easiest last-mile first.

### Option A — paste it into a Code Block (no hosting needed)

Needs a **Business plan or higher**; Squarespace blocks custom code on Personal.

1. Edit the page → add a **Code** block → set the mode to **HTML**.
2. Open `index.html`, copy everything between
   `<!-- ============ JDS RANGE FINDER — COPY FROM HERE ============ -->` and
   `<!-- ============ COPY TO HERE ============ -->`, and paste it in.
3. **Save and view the live page.** Squarespace does not run block JavaScript inside
   the editor preview — in the editor you will see the layout with no chart, which is
   expected and not a bug.

Every style is scoped to `#jds-range-finder`, so nothing leaks into the site theme and
the theme's CSS cannot reach in.

### Option B — iframe it (most robust, works on any plan that allows embeds)

Host `index.html` anywhere static — GitHub Pages off this repo is free:
*Settings → Pages → deploy from branch*, then the file lands at
`https://<user>.github.io/demo-apps/web/range-finder/`.

Then, in a Code block:

```html
<iframe id="jds-frame" src="https://YOUR-HOST/web/range-finder/"
        style="width:100%;height:1900px;border:0;display:block"
        title="Range Finder" loading="lazy"></iframe>
<script>
window.addEventListener("message", function (e) {
  var h = e.data && e.data.jdsRangeFinderHeight;
  if (h) document.getElementById("jds-frame").style.height = h + "px";
});
</script>
```

The page reports its own height, so the frame follows the content instead of leaving a
dead band at the bottom. Drop the script and the fixed height still works — 1900px
covers the phone layout, which is the tallest.

### Option C — a whole page

Serve `index.html` as its own page (GitHub Pages, Netlify, Cloudflare Pages) and link
to it from the nav. Best if you want the estimator to have its own URL to send leads to.

## Design system

The palette and type come from `config/brand.yaml` in the demo repos
(`two-sided-loyalty-demo`, `targeted-offer-policy-demo`, `point-economics-demo`), so a
visitor who clicks from this widget into a case study sees one system rather than two:

| Token | Value | Role |
|---|---|---|
| `--ink` | `#16324A` | text, structure, **observed** data — the number the visitor typed |
| `--surface` | `#FBFAF8` | paper |
| `--estimate` | `#B5175A` | raspberry — **RESERVED** for measured effects, estimates and spreads |
| `--control` | `#93A6B4` | counterfactual and control series |
| `--hairline` / `--grid` | `#C9D4DC` | rules and axes |
| `--muted` | `#5C7288` | captions, secondary text |

The reserved rule is honored the same way the report templates honor it: raspberry appears
only on estimated quantities — the forecast fan and median, the variant posterior, the lift
figures — and never on chrome. Buttons, focus rings, sliders and the hero number wear
`--accent`, which is ink. Type is Inter throughout with a system mono for figures, matching
`report.html.j2`; there is no second display face.

Two deliberate departures, both because this is a live page rather than a printed report:

- **A dark theme exists.** The house system is paper-only. Rather than drop dark support for
  visitors whose OS asks for it, the dark tokens are re-stepped from the *same* hues
  (raspberry `#E5457D`, control `#9FB3C2`) against a deep-ink ground, and validated against
  that surface. Lock it off with `data-jds-theme="light"` if the Squarespace template is
  light-only.
- **`control` keeps its low contrast.** At 2.41:1 on paper it sits below the 3:1 bar for a
  categorical series, which is the point — it means *counterfactual*, and it is supposed to
  recede. That is legal under the relief rule because identity never rests on the color
  alone: there is a legend, a direct label on each curve, and a table twin. The separation
  that matters passes comfortably (CVD ΔE 20.2, normal-vision ΔE 29.5).

## Configuration

At the top of the `<script>` in `index.html`:

```js
var CONFIG = {
  contactUrl: "/contact",   // where "Start a conversation" goes; "" hides the button
  caseStudyUrl: "",         // a published case-study report; the button hides when empty
  paths: 5000,              // simulated futures per run
  draws: 20000,             // posterior draws for the lift test
  horizon: 12               // months forecast
};
```

Set `caseStudyUrl` to the published report from `two-sided-loyalty-demo` (that repo already
builds and publishes `outputs/report` to GitHub Pages in CI) and a second button appears
beside the contact link. The widget is the hook; the case study is the proof.

**Theme.** The widget follows the visitor's OS light/dark setting. If the Squarespace
template is light-only, lock it: `<section id="jds-range-finder" data-jds-theme="light">`
(or `"dark"`).

**Defaults.** The opening state — $250,000/month, +12% growth, 8% volatility, 18%
seasonal swing, and the 8,400/210 vs 8,350/252 test — is deliberately plausible so the
page shows a working model on arrival rather than an empty form. Change the `value`
attributes to move it toward whatever industry you want to attract.

## Capturing the lead

Right now the widget is honest about being a demo: it offers a contact link and a
**Copy this scenario** button that puts a plain-text summary of the visitor's run on
their clipboard. Nothing is transmitted, and there is no form to abandon — which is
the right default for a first prototype.

To collect emails instead, replace the `#jds-copy` button with a Squarespace **Form**
block placed right under the widget, or POST the same summary to a form endpoint:

```js
fetch("https://formspree.io/f/YOUR_ID", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ email: email, scenario: lines.join("\n") })
});
```

The scenario text is already assembled in the `#jds-copy` click handler — that array of
`lines` is exactly what you would want in the notification email.

## Notes

- Accessibility: both charts have a table twin, keyboard readouts (arrow keys move the
  crosshair, Escape dismisses it), visible focus rings, status chips that pair an icon
  with a label rather than relying on color, and a palette validated for colorblind
  separation against both the light and dark surfaces.
- No dependencies. All SVG is generated at the container's real pixel width and
  re-rendered on resize, so axis labels never scale down into mush on a phone.
- The revenue model simulates the assumptions the visitor typed — it is a model of
  their inputs, not a forecast of their business, and the page says so in the fine
  print. Keep that line. It is the difference between a credibility builder and a
  gimmick, and it is also the opening for the actual sales conversation.
- No employer is named anywhere in the copy. The demo repos are careful about this
  ("not affiliated with, sponsored by, or endorsed by any retailer or employer of the
  author"), and the widget matches that posture: it argues from method, not from logos.
