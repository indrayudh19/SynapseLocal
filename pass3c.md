# PASS 3C: Interactive Concept Map (presentation layer only)

The clustering pipeline works well. This pass changes **only how the result is displayed**: a pannable, zoomable, spread-out map where clusters are visible from a distance and detail appears as you zoom in. Nothing upstream changes.

## 0. Operating Rules (read first, obey throughout)

1. **Build, don't deliberate.** Read only the files in section 2. No repo scans, no plan restatement, no summaries before coding.
2. **Additive only.** New files and one new stage. Stages 1-3, their outputs and the existing PNG stay exactly as they are. If something is not required by this spec, do not touch it.
3. **No new Python dependencies.** The map is one self-contained HTML file: inline SVG + vanilla JavaScript. No CDN, no JS library, no network requests.
4. **No model calls.** Stage 4 uses no Ollama, no embedding model, no sklearn. numpy and the standard library only (networkx is allowed for the spanning tree).
5. **One stage at a time:** write Stage 4 (Python), then the HTML template, then the UI wiring. Finish each before starting the next.
6. **Verify statically only** (section 9). Do not launch browsers, take screenshots or run benchmarks. The user checks the visuals.

---

## 1. Goal and Architecture

```
concept_clusters.json  (Stage 3, unchanged)
concept_graph.json     (Stage 2, read-only, for evidence sentences)
        |
        v
[Stage 4] BUILD INTERACTIVE MAP  (numpy + json only)
        -> concept_map.html  (single file, inline data, no external resources)
        |
        v
Concept Clusters tab: st.components.v1.html(...)  +  download button
```

- Stage 4 is one more independent service: `run(session_id) -> dict`, its own button, no auto-chaining, same `.running` lock file and content-hash cache as the other stages.
- The existing PNG (from Stage 3) remains: it is the **fallback** shown when the interactive map has not been built, and an optional static export.
- Why this is light: coordinates are precomputed in Python, so the browser runs no physics simulation. Pan/zoom is a single SVG transform. Payload is a small JSON blob (target < 1 MB total HTML).

**Semantic zoom, decided per cluster by its on-screen size** (so a big cluster expands sooner than a small one):

| Cluster on-screen diameter | What is visible |
|---|---|
| < `EXPAND_PX` (240) | Collapsed: a translucent colored bubble (halo) with the hub title. Members hidden. |
| `EXPAND_PX` to `DETAIL_PX` (640) | Expanded: hub circle, member dots, labels for the top 5 members, bridge lines. |
| >= `DETAIL_PX` | Detail: all member labels that fit, relation labels on links, full hover info. |

---

## 2. Files

Read once: `representation/paths.py` (to append one helper), `app.py` (Concept Clusters tab only), and the **shape** of `concept_clusters.json` and `concept_graph.json` (peek at one existing session file, do not read code).

**Protected (do not modify or import from):** `cluster.py`, `concept_plot.py`, `concept_schema.py`, `extract_concepts.py`, `build_concept_graph.py`, `embed_store.py`, all pass1 code, the chat tab and all inference code, session management.

New files:

```
representation/
  map_layout.py     # Stage 4: layout + data assembly + HTML write
  map_template.html # self-contained template (CSS + JS), placeholder for data
paths.py            # append: concept_map_html_path(session_id)
```

Constants live at the top of `map_layout.py` (not in `concept_schema.py`).

---

## 3. Stage 4: Layout + Export (`map_layout.py`)

`run(session_id) -> dict`

**Input:** `concept_clusters.json` (structure, links, bridges, metrics) and `concept_graph.json` (read-only: `best_sentence` per concept, top relations with `evidence`). Match members to concepts by exact `name`; if a name is missing, skip the extra info for that member.

**Output:** `concept_map.html` in the session's `representation/` folder. Return `{n_clusters, n_members_drawn, n_links, html_kb}`.

**Gate:** if `status != "ok"`, return the same status and write nothing (the UI falls back to the message/PNG).

**Cache:** hash of the input files' content + constants; if unchanged and the HTML exists, return immediately.

### 3.1 Constants

```python
EXPAND_PX, DETAIL_PX = 240, 640
MAX_MEMBERS_PER_CLUSTER, MAX_MEMBERS_TOTAL = 60, 1500   # rest summarized as "+k more" in the panel
SPREAD = 1.4              # global spacing multiplier (the map must feel spread out)
HALO_GAP = 140            # min world-unit gap between cluster halos
HUB_R_BASE, HUB_R_PER = 34, 6          # hub radius = base + per*sqrt(n)
RING_START, RING_STEP, ARC_SPACING = 56, 52, 36
PALETTE = ["#e8703a","#3b8fd9","#4cae4f","#d94b4b","#9b6fd1","#e0a458",
           "#8c6d62","#5fb3b3","#d96aa7","#a0b2d6","#b5c94f","#7a7f8c"]
```

### 3.2 Per-cluster internal layout (concentric rings, so large clusters do not jam one ring)

1. Sort members by `importance` descending; cap to `MAX_MEMBERS_PER_CLUSTER` (and a global cap of `MAX_MEMBERS_TOTAL`, trimming the largest clusters first).
2. Hub circle radius `HUB_R_BASE + HUB_R_PER*sqrt(n)`.
3. Ring `k` has radius `r_k = hubR + RING_START + k*RING_STEP` and capacity `floor(2*pi*r_k / ARC_SPACING)`. Fill ring 0 first, then ring 1, and so on, by importance. Offset each ring's start angle by `k*0.35` rad to avoid radial alignment.
4. **Bridge members** (from `bridges`) are placed on ring 0 at the free slot closest to the direction of their partner cluster's center (compute after step 3.3 centers; fall back to normal slots if ring 0 is full).
5. Member dot radius `5 + 9*importance` (world units). Halo radius = outermost ring radius + 70.
6. Record `rank` (0 = most important) per member, used for label priority.

### 3.3 Cluster placement

1. **Classical MDS on cluster affinity** (numpy only): `D_ab = 1/(M_ab + 0.05)` using the stored cluster `links[].affinity`; missing pairs get `1.5 * max finite D`; `B = -0.5 * J D^2 J`; take the top 2 eigenvectors scaled by `sqrt(max(eigenvalue, 0))`. One cluster: origin. Two: a horizontal line. Degenerate case: seeded jitter (`seed=42`).
2. Multiply all centers by `SPREAD`.
3. **Relaxation (<= 300 iterations):** while any two halos are closer than `r_a + r_b + HALO_GAP`, push them apart along the line between centers (equal split). This guarantees no overlap and a spread-out canvas.
4. Compute bounds and a `viewBox` with a 200-unit margin. Stage 4 does not decide the initial zoom; the JS fits the map to the viewport.

### 3.4 Links (reduce the spaghetti)

- Draw a **maximum spanning tree** over cluster affinity (`networkx.maximum_spanning_tree`; guard for disconnected graphs by treating each component separately), plus any additional link with affinity >= the median affinity. Mark `tree: true` on the tree edges.
- Each link carries its strongest underlying edge `{src, rel, tgt}` for its label.
- Hub-member edges: solid if `link_to_hub` starts with `direct`, otherwise dotted.

### 3.5 Assemble and write

Build one dict (schema below), serialize with `json.dumps(..., ensure_ascii=False, separators=(",",":"))`, and **sanitize for embedding in a `<script type="application/json">` element**: replace `<` with `\u003c`, `>` with `\u003e`, `&` with `\u0026`, and U+2028/U+2029 with their escapes. Concept names come from document text, so this is mandatory. Insert at the template placeholder `/*__MAP_DATA__*/`, write UTF-8.

```json
{"meta":{"title":"Concept Map","metrics":{"clusters":0,"concepts":0,"modularity":0,"silhouette":0},
         "globalTopics":[],"expandPx":240,"detailPx":640},
 "clusters":[{"id":0,"title":"microservices","x":0,"y":0,"hubR":52,"halo":260,"color":"#e8703a",
              "n":27,"drawn":27,"keywords":["..."],"sourceMix":{"slides.pptx":0.6,"notes.pdf":0.4}}],
 "members":[{"id":"c0m3","c":0,"name":"API Gateway","kind":"component","x":0,"y":0,"r":9,
             "imp":0.62,"mem":0.84,"direct":true,"multi":true,"src":["slides.pptx","notes.pdf"],"rank":2,
             "sent":"<=200 chars","rels":[{"t":"service discovery","r":"uses","ev":"<=160 chars"}]}],
 "links":[{"a":0,"b":1,"w":0.55,"rel":"uses","src":"IaaS","tgt":"hardware","tree":true}],
 "bridges":[{"m":"c0m5","to":1,"ratio":0.7}]}
```

`rels`: up to 3 relations involving that concept from `concept_graph.json`, highest support first.

End of stage: release memory, return stats. No model unload needed (none used).

---

## 4. HTML Template (`map_template.html`)

One file, no external resources. Dark theme matching the app (background `#05060a`, text `#e6e6ef`, monospace stack). Add at the top of `<head>`:

```html
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:">
```

Data is read from `<script id="map-data" type="application/json">/*__MAP_DATA__*/</script>` via `JSON.parse(textContent)`.

### 4.1 Structure

- Full-size `<svg>` (`touch-action:none`) with one `<g id="viewport">` holding, in order: halos, links, hub-member edges, bridge lines, member dots, hub circles, labels. One `<g class="cluster">` per cluster for halo, members and edges so LOD classes toggle per cluster.
- HTML overlays (absolutely positioned): top-left info chip (title + metrics), top-right search box, bottom-left zoom controls (`+`, `-`, reset), bottom-right minimap, right-side detail panel (hidden until a selection), and a collapsible legend behind a `?` button.
- Stroke widths use `vector-effect: non-scaling-stroke` so lines keep a constant screen thickness at any zoom.

### 4.2 Rendering rules

- **Safety:** all text goes in with `textContent` / `createTextNode`. Never use `innerHTML`, `eval`, `new Function` or inline event attributes with data.
- **Colors:** hub and members take the cluster color; member dot `fill-opacity = 0.35 + 0.65*mem`; gold stroke `#f2c14e` (width 2) if `multi` (supported by 2+ source files); dotted hub-member edge if not `direct`; dashed gray line from a bridge member to its partner hub; hub-hub links purple `#8b5cf6`, width `0.8 + 0.6*ln(1 + w*10)`, drawn behind everything, straight, endpoints on halo edges.
- **Labels are counter-scaled so their on-screen size is constant.** Set CSS variable `--inv = 1/k` on the SVG root on every zoom and give labels `font-size: calc(var(--fs) * var(--inv))` (`--fs` in px, 12px members, 14px hub circle, 18px collapsed halo titles). Position each label with `<g transform="translate(x ± r, y)">` and `<text dx="±0.6em">`, so the gap stays constant on screen. Anchor `start` for the right half of the hub, `end` for the left half. If `calc()` on SVG font-size misbehaves in a browser, fallback is to set the `font-size` attribute on visible labels during the layout pass.
- **Labels never wrap**, so words are never split. Truncate to 28 characters with an ellipsis; full name is in the tooltip and panel.

### 4.3 Semantic zoom and label collision

- On every transform change (rAF-throttled), for each cluster compute `diameterPx = 2*halo*k`. Toggle classes: `collapsed` (< `expandPx`), `expanded`, `detail` (>= `detailPx`). CSS controls visibility with a short opacity transition (about 150 ms). Clusters fully outside the viewport get `display:none`.
- Link relation labels appear only if the link's on-screen length exceeds 300 px, and `uses` labels only in `detail`.
- **Label layout pass, debounced 80 ms after the last pan/zoom event:** build candidates (hub titles first, then selected/search/neighbor nodes, then members by `rank`), estimate each box in screen space (`chars * 0.58 * fontPx` by `fontPx * 1.3`), place greedily, and hide any label that overlaps an already accepted one (use a coarse grid hash for speed). Expanded clusters allow only `rank < 5`; detail clusters allow all.

### 4.4 Interaction

- **Wheel** zooms about the cursor; **drag** pans (pointer events); **pinch** zooms (two pointers); **double-click** zooms in 2x at the point.
- Zoom limits: `0.8*k0` to `14*k0`, where `k0` is the scale at which the whole map fits the viewport (computed on load and on resize; initial view = fit-all with 6% padding).
- **Click a hub or halo:** animated fly-to (about 400 ms ease-in-out) fitting that cluster to ~70% of the viewport. **Click a member:** select it. **Click empty space or press Esc:** clear selection and fly back out is not required, only clear.
- **Selection:** dim everything except the selected node, its hub, its direct links and its bridge partner (CSS classes on the root, no re-render). Show the detail panel.
- **Detail panel** (right side, 300 px, closable): member name, kind, cluster title, source files, importance and membership percentages, best sentence, up to 3 relations with evidence. For a hub: cluster size, `drawn`/`+k more`, keywords, source mix bar, and top links to other clusters with relation labels.
- **Hover:** native-style tooltip with name and kind. No layout work on hover.
- **Search:** case-insensitive substring over member and hub names; dropdown of up to 8 matches; Enter or click flies to the node, selects it and keeps the label visible. Empty or no match is handled without errors.
- **Keyboard:** `+`/`-` zoom, `0` reset, `Esc` clear.
- **Minimap** (about 160x110 px): cluster halos as colored circles plus the current viewport rectangle; click or drag on it recenters the main view.
- **Reset button:** returns to fit-all.
- Handle window resize and zero-cluster / one-cluster data gracefully (no console errors).

### 4.5 Size and performance targets

Single file under about 1 MB for typical sessions; no per-frame DOM creation; during pan/zoom only the viewport transform, the `--inv` variable and per-cluster class toggles change.

---

## 5. UI Wiring (Concept Clusters tab only)

1. Add a **step 4 button: "Build interactive map"**, enabled only when `concept_clusters.json` exists with `status == "ok"`. Status text derives from `concept_map.html` existing. Same pattern as the other steps: `st.spinner`, no auto-chaining, no "run all".
2. **Display logic:**
   - If `concept_map.html` exists: `st.components.v1.html(html_text, height=820, scrolling=False)`, a one-line caption ("Scroll to zoom, drag to pan, click a hub to focus, search to jump"), and `st.download_button("Download interactive map (.html)", data=html_text, file_name="concept_map.html", mime="text/html")` so it can be opened full screen in a browser tab.
   - Put the existing PNG inside an expander labeled "Static image (PNG)".
   - If the HTML does not exist: behave exactly as before (show the PNG).
3. Keep all existing metric cards, tables, expanders and steps 1-3 unchanged.
4. **Do not add widgets to this tab that trigger reruns while the map is shown** (a Streamlit rerun recreates the iframe and resets the view). Read the HTML file once per render; no sliders or selectboxes near the map.
5. No changes to `requirements.txt`.

---

## 6. Constraints Recap

- Offline, self-contained, no external requests; no new Python or JS dependencies.
- One stage at a time, own button, same lock-file and hash-cache conventions.
- Upstream stages, the PNG renderer and all other tabs untouched.
- The iframe is one-way: clicks inside the map do not call back into Python (out of scope).

---

## 7. Known Limits (do not add machinery for these)

- Streamlit reruns reset the map view.
- SVG is intended for up to about 1,500 drawn members (hence the caps); canvas rendering is out of scope.
- Very large clusters show their top `MAX_MEMBERS_PER_CLUSTER` members; the panel reports `+k more`.
- Sub-cluster splitting of oversized clusters is a clustering change and out of scope for this pass.

---

## 8. Build Order

1. `paths.py` helper -> `map_layout.py` (3.1-3.5), finished and saved.
2. `map_template.html` (section 4), finished and saved.
3. UI wiring (section 5).

---

## 9. Done Criteria (static checks only)

- [ ] `map_layout.run(session_id)` on an existing session writes `concept_map.html`; the embedded JSON parses; file size reported (target < 1 MB).
- [ ] Grep of the HTML: no `http://` or `https://` other than the SVG namespace `http://www.w3.org/2000/svg`; no `innerHTML`, `eval`, `new Function`.
- [ ] Cluster halos do not overlap in the exported coordinates (assert minimum gap >= `HALO_GAP` in the stage).
- [ ] The Concept Clusters tab shows the map and download button when the HTML exists, and the PNG as before when it does not.
- [ ] Stages 1-3, `cluster.py`, `concept_plot.py`, the chat tab and `requirements.txt` are unchanged (diff shows only the files listed in section 2).

If a check fails, fix that specific issue and stop.

---

## 10. Final Reminder to the Agent

Presentation only. One new stage, one HTML template, one UI addition. Precompute the layout in Python, render with inline SVG, zoom semantically per cluster, keep labels readable and never split a word. Touch nothing upstream, add no dependencies, and spend no tokens on extra analysis.
