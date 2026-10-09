"""Standalone visual layout editor for on-demand event plots."""

import html
import json
import re
from urllib.parse import quote


def editor_content(event: dict, series: dict[str, str], epochs: dict[str, list[str]], metadata: dict,
                   channels: dict[str, list[str]]) -> tuple[str, str]:
    """Return HTML body and styles for the dedicated plot editor page."""
    data = json.dumps(series, ensure_ascii=False).replace("<", "\\u003c")
    times = json.dumps(epochs, ensure_ascii=False).replace("<", "\\u003c")
    channel_json = json.dumps(channels, ensure_ascii=False).replace("<", "\\u003c")
    event_path = json.dumps(event["path"], ensure_ascii=False).replace("<", "\\u003c")
    from flare_metadata import scientific_caption
    heading, subtitle = scientific_caption(metadata, list(series)[:1])
    heading_json = json.dumps(heading, ensure_ascii=False).replace("<", "\\u003c")
    subtitle_json = json.dumps(subtitle, ensure_ascii=False).replace("<", "\\u003c")
    location_json = json.dumps({"x": metadata["x"], "y": metadata["y"], "flare": metadata["class"],
                                "start": metadata["start"], "peak": metadata["peak"], "end": metadata["end"]})
    name = html.escape(event["name"])
    body = r"""
    <header class="studio-heading"><div><a href="__EVENT_URL__" class="studio-back">← Back to event</a>
      <h1>Plot studio</h1><p>__EVENT_NAME__ · Drag panels by their headers and resize from the lower-right corner.</p></div></header>
    <div class="studio-layout">
      <section class="studio-workspace" aria-label="Layout preview">
             <div class="studio-toolbar"><strong>Layout preview</strong><span>Panel layout, event markers and observation time are previewed here; actual data appears after generation.</span>
            <button type="button" class="button" id="addPlotPanel">+ Add panel</button></div>
        <div class="studio-scroll"><div id="plotCanvas" class="plot-canvas" aria-label="Layout preview">
            <strong id="canvasTitle" class="canvas-title">__HEADING__</strong>
            <span id="canvasSubtitle" class="canvas-subtitle">__SUBTITLE__</span>
            <span id="canvasLegend" class="canvas-legend"></span>
          <div id="plotStage" class="plot-stage" aria-label="Drag and resize plot panels"></div></div></div>
      </section>
      <aside class="studio-sidebar">
         <h2>Plot settings</h2>
          <label>Figure template <select id="figureTemplate">
            <option value="overview">Flare overview · map + Sun + flux</option>
            <option value="irradiance">Solar irradiance · Sun + GOES + SOHO</option>
            <option value="response">Ionospheric response · map + index + flux</option>
            <option value="comparison">Map comparison · two products</option>
            <option value="timeline">Time series · stacked panels</option>
            <option value="custom">Custom layout</option>
          </select></label>
          <p class="studio-note" id="templateHint">Choose a template, then drag, resize or edit panels to customize it.</p>
          <label>Plot style <select id="plotStyle"><option value="simple" selected>Simple · white, fine grid</option>
            <option value="plotter">Plotter · colored axes</option></select></label>
          <label>Observation time (UTC) <select id="observationTime"></select></label>
          <p class="studio-note" id="observationHint">One time for all maps and time-series panels, from the available HDF5 datasets.</p>
         <label>Additional caption <input id="plotTitle" maxlength="100" placeholder="Optional description"></label>
        <label>Email identifier <input id="plotEmail" type="email" required autocomplete="email" placeholder="name@example.com"></label>
        <p class="studio-note">Use the same email in the catalog to find your plots. No messages are sent.</p>
        <div id="panelSettings" class="panel-settings">
          <h3>Selected panel</h3>
          <label>Data series <select id="panelSeries"></select></label>
           <label id="colorField">Line color <input id="panelColor" type="color" value="#2878a5"></label>
             <p id="mapPalette" class="studio-note" hidden>Fixed map scales: ROTI · viridis 0–0.5 TECu/min; dTEC · RdBu_r −0.5…0.5 TECu.</p>
          <div class="size-fields"><label>Width (%) <input id="panelWidth" type="number" min="16" max="100" step="1"></label>
            <label>Height (%) <input id="panelHeight" type="number" min="16" max="100" step="1"></label></div>
          <button type="button" class="button" id="removePanel">Remove selected panel</button>
        </div>
        <button type="button" class="button studio-generate" id="generatePlot">Generate plot</button>
        <div id="plotMessage" role="status" aria-live="polite"></div>
        <div id="renderedPlot" hidden><a id="renderedLink" target="_blank" rel="noopener">Open full-size plot</a>
          <img id="renderedImage" alt="Generated plot"></div>
      </aside>
    </div>
    <script>
    (() => {
      const series = __SERIES_JSON__;
      const epochs = __EPOCHS_JSON__;
      const channels = __CHANNELS_JSON__;
      const eventPath = __EVENT_JSON__;
      const heading = __HEADING_JSON__;
      const subtitle = __SUBTITLE_JSON__;
      const location = __LOCATION_JSON__;
      const canvas = document.getElementById('plotStage');
      const message = document.getElementById('plotMessage');
       const select = document.getElementById('panelSeries');
       const style = document.getElementById('plotStyle');
      const color = document.getElementById('panelColor');
       const epoch = document.getElementById('observationTime');
      const width = document.getElementById('panelWidth');
      const height = document.getElementById('panelHeight');
       const panels = [];
       let selected = null;
       const templatePicker = document.getElementById('figureTemplate');
       const templateHint = document.getElementById('templateHint');
       const has = (...keys) => keys.every(key => key in series);
       const indexKey = ['roti:isfai_index', 'roti:gsflai_index', 'roti:day_night_index',
         ...Object.keys(series).filter(key => key.includes(':') && !key.startsWith('map:'))].find(key => key in series);
       const fluxKey = has('goes') ? 'goes' : has('soho') ? 'soho' : null;
       const maps = Object.keys(series).filter(key => key.startsWith('map:') && (epochs[key] || []).length);
       const mapPair = maps.flatMap((left, i) => maps.slice(i + 1).map(right => [left, right]))
         .find(pair => (epochs[pair[0]] || []).some(time => (epochs[pair[1]] || []).includes(time)));
       const timelineKeys = ['goes', 'soho', indexKey].filter(key => key && key in series);
       const templates = {
         overview: has('map:roti', 'sun', 'goes', 'soho') && (epochs['map:roti'] || []).length ? [
           ['map:roti', .02, .02, .58, .39], ['sun', .64, .02, .34, .39],
           ['goes', .02, .44, .96, .25], ['soho', .02, .72, .96, .26]] : null,
         irradiance: has('sun', 'goes', 'soho') ? [
           ['sun', .02, .02, .30, .46], ['goes', .35, .02, .63, .46],
           ['soho', .02, .53, .96, .42]] : null,
         response: maps.length && indexKey && fluxKey ? [
           [maps[0], .02, .02, .58, .46], [indexKey, .64, .02, .34, .46],
           [fluxKey, .02, .53, .96, .42]] : null,
         comparison: mapPair ? [
           [mapPair[0], .02, .08, .47, .80], [mapPair[1], .51, .08, .47, .80]] : null,
         timeline: timelineKeys.length >= 2 ? timelineKeys.map((key, i) =>
           [key, .02, timelineKeys.length === 2 ? .02 + i * .50 : .02 + i * .32,
            .96, timelineKeys.length === 2 ? .45 : .29]) : null,
       };
       for (const option of templatePicker.options) if (option.value !== 'custom') {
         option.disabled = !templates[option.value];
       }
       function markCustom() {
         templatePicker.value = 'custom';
         templateHint.textContent = text('Custom layout · your changes are kept until you choose another template.',
           'Своя компоновка · изменения сохраняются, пока вы не выберете другой шаблон.');
       }
       function applyTemplate(name) {
         const layout = templates[name];
         if (!layout) return;
         panels.length = 0;
         for (const [key, x, y, w, h] of layout) panels.push({
           id: String(Date.now()) + Math.random(), series: key, color: defaultColor(key), customColor: false,
           rect: {x, y, w, h},
         });
         selected = panels[0].id;
         templatePicker.value = name;
         templateHint.textContent = text('Drag, resize or edit any panel to make this template your own.',
           'Перетащите, измените размер или данные любой панели, чтобы настроить шаблон под себя.');
         render();
       }
       templatePicker.onchange = () => {
         if (templatePicker.value !== 'custom') applyTemplate(templatePicker.value);
         else markCustom();
       };
       document.getElementById('plotTitle').addEventListener('input', updateCaption);
        function availableTimes() {
          const maps = [...new Set(panels.filter(panel => panel.series.startsWith('map:')).map(panel => panel.series))];
          if (!maps.length) return [...new Set(Object.values(epochs).flat())].sort();
          return (epochs[maps[0]] || []).filter(time => maps.every(key => (epochs[key] || []).includes(time)));
        }
        function syncEpoch() {
          const available = availableTimes();
          const previous = epoch.value;
          epoch.replaceChildren();
          for (const time of available) {
            const option = document.createElement('option'); option.value = time;
            option.textContent = time + ' UTC'; epoch.append(option);
          }
           const peak = Date.parse(location.peak);
           const nearest = Number.isFinite(peak) && available.length ? available.reduce((best, time) =>
             Math.abs(Date.parse(time.replace(' ', 'T') + 'Z') - peak) <
             Math.abs(Date.parse(best.replace(' ', 'T') + 'Z') - peak) ? time : best) :
             available[Math.floor(available.length / 2)];
           epoch.value = available.includes(previous) ? previous : nearest || '';
          epoch.disabled = !available.length;
          document.getElementById('observationHint').textContent = available.length ?
            text('One time for all maps and time-series panels, from the available HDF5 datasets.',
              'Одно время для всех карт и временных рядов, из доступных данных HDF5.') :
            text('No common observation time for these maps.', 'У выбранных карт нет общего времени наблюдения.');
        }
       function updateCaption() {
          document.getElementById('canvasTitle').textContent = heading;
          const peak = Date.parse(location.peak);
          const selectedTime = epoch.value ? Date.parse(epoch.value.replace(' ', 'T') + 'Z') : NaN;
          const reference = epoch.value && selectedTime !== peak ?
            `Observation: ${epoch.value.slice(11, 16)} UTC` : '';
          document.getElementById('canvasSubtitle').textContent = [subtitle, reference,
            document.getElementById('plotTitle').value].filter(Boolean).join('  ·  ');
          const markers = location.peak && reference ?
            [`<span style="color:#bd4651">┊</span> Flare peak`,
             `<span style="color:#9a6400">⋮</span> Selected time ${epoch.value.slice(11, 16)} UTC`] : [];
         document.getElementById('canvasLegend').innerHTML = panels.some(panel =>
           panel.series !== 'sun' && !panel.series.startsWith('map:')) ? markers.join('　') : '';
       }
       const text = (en, ru) => document.documentElement.lang === 'ru' ? ru : en;
       const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
       function previewTicks() {
         if (!location.start || !location.end) return 'UTC';
         const start = Date.parse(location.start), end = Date.parse(location.end);
         if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return 'UTC';
          const peak = Date.parse(location.peak);
          const middle = Number.isFinite(peak) && peak >= start && peak <= end ? peak : (start + end) / 2;
          const half = Number.isFinite(peak) ? 15 * 60000 : Math.max(15 * 60000, (end - start) / 2 + 5 * 60000);
         const label = stamp => new Date(stamp).toISOString().slice(11, 16);
          const moment = Date.parse(epoch.value.replace(' ', 'T') + 'Z');
          const left = Number.isFinite(moment) ? Math.min(middle - half, moment - 5 * 60000) : middle - half;
          const right = Number.isFinite(moment) ? Math.max(middle + half, moment + 5 * 60000) : middle + half;
          return `${label(left)}     ${label((left + right) / 2)}     ${label(right)}`;
       }
      function canPlace(current, rect) {
        return rect.x >= 0 && rect.y >= 0 && rect.w >= .16 && rect.h >= .16 &&
          rect.x + rect.w <= 1.000001 && rect.y + rect.h <= 1.000001 &&
          panels.every(other => other === current || rect.x >= other.rect.x + other.rect.w - 1e-6 ||
            other.rect.x >= rect.x + rect.w - 1e-6 || rect.y >= other.rect.y + other.rect.h - 1e-6 ||
            other.rect.y >= rect.y + rect.h - 1e-6);
      }
       function place(panel, rect) {
         if (!canPlace(panel, rect)) return false;
         if (Object.keys(rect).some(axis => rect[axis] !== panel.rect[axis])) markCustom();
         panel.rect = rect; return true;
      }
        const defaultColor = key => (style.value === 'simple' ?
          {goes: (channels.goes || []).includes('xrsa') ? '#d1495b' : '#111111',
            soho: '#006400',
           day_night_index: '#333333', gsflai_index: '#006400', isfai_index: '#333333'} :
         {goes: '#d1495b', soho: '#2878a5', day_night_index: '#2878a5', gsflai_index: '#2a9d8f', isfai_index: '#e76f51'}
       )[key.split(':').at(-1)] || (style.value === 'simple' ? '#333333' : '#2878a5');
       style.onchange = () => {
         document.getElementById('plotCanvas').dataset.style = style.value;
         for (const panel of panels) if (!panel.customColor) panel.color = defaultColor(panel.series);
         render();
       };
       document.getElementById('plotCanvas').dataset.style = style.value;
      for (const [key, label] of Object.entries(series)) {
        const option = document.createElement('option'); option.value = key; option.textContent = label; select.append(option);
      }
      function syncSettings() {
        const panel = panels.find(item => item.id === selected);
        document.getElementById('panelSettings').hidden = !panel;
        if (!panel) return;
        select.value = panel.series; color.value = panel.color;
        const map = panel.series.startsWith('map:');
         document.getElementById('colorField').hidden = map || panel.series === 'sun';
        document.getElementById('mapPalette').hidden = !map;
        width.value = Math.round(panel.rect.w * 100); height.value = Math.round(panel.rect.h * 100);
        canvas.querySelectorAll('.canvas-panel').forEach(card => card.classList.toggle('active', card.dataset.id === selected));
      }
       function position(card, rect) {
         card.style.left = `${rect.x * 100}%`; card.style.top = `${rect.y * 100}%`;
         card.style.width = `${rect.w * 100}%`; card.style.height = `${rect.h * 100}%`;
         card.dataset.narrow = rect.w < .55;
      }
      function render() {
        syncEpoch();
        canvas.replaceChildren();
        for (const panel of panels) {
          const card = document.createElement('div'); card.className = 'canvas-panel'; card.dataset.id = panel.id;
          card.tabIndex = 0; card.setAttribute('aria-label', `${series[panel.series]}, drag to move, arrows to nudge`);
          const bar = document.createElement('div'); bar.className = 'canvas-panel-bar';
            const caption = document.createElement('strong'); caption.textContent =
               (panel.series.startsWith('map:') ? `Global ${panel.series.slice(4).toUpperCase().replace('DTEC', 'dTEC')} map` : series[panel.series]) +
               (panel.series.startsWith('map:') && epoch.value ? ' · ' + epoch.value.slice(11, 16) + ' UTC' : '');
           const grip = document.createElement('span'); grip.textContent = String.fromCharCode(65 + panels.indexOf(panel)) + ' ⠿';
           grip.setAttribute('aria-hidden', 'true');
          bar.append(grip, caption);
           const chart = document.createElement('div'); chart.className = 'canvas-chart';
          const xLabel = document.createElement('span'); xLabel.className = 'canvas-xlabel';
          const yLabel = document.createElement('span'); yLabel.className = 'canvas-ylabel';
           const ticks = document.createElement('span'); ticks.className = 'canvas-ticks';
           const units = document.createElement('span'); units.className = 'canvas-units';
           if (panel.series === 'sun') {
             chart.classList.add('canvas-sun');
             const x = location.x === null ? null : 100 + location.x / 960 * 38;
             const y = location.y === null ? null : 40 - location.y / 960 * 38;
             chart.innerHTML = '<svg viewBox="0 0 200 80" aria-hidden="true"><circle cx="100" cy="40" r="38" fill="#f6c85f" stroke="#e6a23c"/>' +
               (x === null ? '' : `<text x="${x}" y="${y}" fill="#e76f51" stroke="white" stroke-width=".4" font-size="16">★</text>`) + '</svg>';
             ticks.textContent = x === null ? text('Flare position unavailable', 'Положение вспышки неизвестно') :
               `HPC ${Math.round(location.x)}″, ${Math.round(location.y)}″`;
           } else if (panel.series.startsWith('map:')) {
            chart.classList.add('canvas-map');
            chart.innerHTML = '<svg viewBox="0 0 200 80" preserveAspectRatio="none" aria-hidden="true"><circle cx="40" cy="35" r="7"/><circle cx="77" cy="46" r="5"/><circle cx="112" cy="20" r="6"/><circle cx="155" cy="40" r="8"/></svg>';
            xLabel.textContent = 'Longitude'; yLabel.textContent = 'Latitude'; ticks.textContent = '−180     0      180';
            const scale = document.createElement('span'); scale.className = 'canvas-scale';
              scale.textContent = panel.series === 'map:roti' ? '0 → 0.5 TECu/min' : '−0.5 → 0.5 TECu';
            scale.style.borderImage = panel.series === 'map:roti' ?
              'linear-gradient(#fde725, #21918c, #440154) 1' : 'linear-gradient(#b2182b, #f7f7f7, #2166ac) 1';
            card.append(scale);
           } else {
             chart.innerHTML = '<svg viewBox="0 0 200 80" preserveAspectRatio="none" aria-hidden="true"><polyline points="0,64 28,52 43,60 60,35 83,50 105,17 125,37 145,22 165,48 200,18"/></svg>';
             if (panel.series === 'goes' && (channels.goes || []).includes('xrsa')) {
               const line = document.createElement('span'); line.className = 'canvas-second-line';
               line.textContent = 'XRS-A (0.05–0.4 nm) · XRS-B (0.1–0.8 nm)'; chart.append(line);
              } else if (panel.series === 'soho' && (channels.soho || []).includes('flux_26_34')) {
                chart.querySelector('svg').insertAdjacentHTML('beforeend',
                  '<polyline class="canvas-secondary" points="0,54 28,53 43,48 60,53 83,51 105,44 125,50 145,43 165,45 200,42"/>');
                const line = document.createElement('span'); line.className = 'canvas-second-line';
                line.textContent = '26–34 nm — — · 0.1–50 nm ━'; chart.append(line);
             }
            const column = panel.series.split(':')[1];
              units.textContent = panel.series === 'goes' || panel.series === 'soho' ?
                'Flux change (%)' : ({day_night_index: 'Day/night', gsflai_index: 'GSFLAI', isfai_index: 'ISFAI'}[column] || column);
               xLabel.textContent = 'Time (UTC)'; ticks.textContent = previewTicks();
             if (location.start && location.end) {
               const start = Date.parse(location.start), end = Date.parse(location.end);
                const peak = Date.parse(location.peak);
                const middle = Number.isFinite(peak) && peak >= start && peak <= end ? peak : (start + end) / 2;
                const half = Number.isFinite(peak) ? 15 * 60000 : Math.max(15 * 60000, (end - start) / 2 + 5 * 60000);
                const moment = Date.parse(epoch.value.replace(' ', 'T') + 'Z');
                const left = Number.isFinite(moment) ? Math.min(middle - half, moment - 5 * 60000) : middle - half;
                const right = Number.isFinite(moment) ? Math.max(middle + half, moment + 5 * 60000) : middle + half;
                for (const [stamp, kind] of [[location.peak, 'peak'],
                                             [Date.parse(epoch.value.replace(' ', 'T') + 'Z') === Date.parse(location.peak) ? '' : epoch.value, 'map']]) {
                 const value = Date.parse(kind === 'map' ? stamp.replace(' ', 'T') + 'Z' : stamp);
                 if (!Number.isFinite(value)) continue;
                  const x = (value - left) / (right - left);
                 if (x < 0 || x > 1) continue;
                 const line = document.createElement('span'); line.className = 'canvas-marker ' + kind;
                 line.style.left = `${x * 100}%`; chart.append(line);
               }
             }
           }
           chart.style.color = panel.color;
          const handle = document.createElement('div'); handle.className = 'resize-handle';
          handle.title = text('Drag to resize', 'Потяните для изменения размера');
          handle.setAttribute('aria-label', 'Resize panel');
           card.append(bar, chart, units, ticks, xLabel, yLabel, handle); position(card, panel.rect);
          card.addEventListener('pointerdown', () => { selected = panel.id; syncSettings(); });
          card.addEventListener('keydown', event => {
            const moves = {ArrowLeft: [-.01, 0], ArrowRight: [.01, 0], ArrowUp: [0, -.01], ArrowDown: [0, .01]};
            if (!moves[event.key]) return;
            event.preventDefault(); const [dx, dy] = moves[event.key];
             place(panel, {...panel.rect, x: clamp(panel.rect.x + dx, 0, 1 - panel.rect.w),
               y: clamp(panel.rect.y + dy, 0, 1 - panel.rect.h)}); position(card, panel.rect);
          });
          function drag(event, resize) {
            event.preventDefault(); selected = panel.id; syncSettings();
            const startX = event.clientX, startY = event.clientY, initial = {...panel.rect};
            const target = event.currentTarget;
            target.setPointerCapture(event.pointerId);
            const move = next => {
              const dx = (next.clientX - startX) / canvas.clientWidth;
              const dy = (next.clientY - startY) / canvas.clientHeight;
              if (resize) {
                 place(panel, {...initial, w: clamp(initial.w + dx, .16, 1 - initial.x),
                   h: clamp(initial.h + dy, .16, 1 - initial.y)});
               } else {
                 place(panel, {...initial, x: clamp(initial.x + dx, 0, 1 - initial.w),
                   y: clamp(initial.y + dy, 0, 1 - initial.h)});
              }
              position(card, panel.rect);
              width.value = Math.round(panel.rect.w * 100); height.value = Math.round(panel.rect.h * 100);
            };
            const end = () => { target.removeEventListener('pointermove', move); target.removeEventListener('pointerup', end); target.removeEventListener('pointercancel', end); };
            target.addEventListener('pointermove', move);
            target.addEventListener('pointerup', end);
            target.addEventListener('pointercancel', end);
          }
          bar.addEventListener('pointerdown', event => drag(event, false));
          handle.addEventListener('pointerdown', event => { event.stopPropagation(); drag(event, true); });
          canvas.append(card);
        }
         syncSettings();
         updateCaption();
        document.getElementById('addPlotPanel').disabled = panels.length >= 6;
      }
      function addPanel() {
        if (panels.length >= 6 || !Object.keys(series).length) return;
        const index = panels.length;
         let rect;
         for (let row = 0; row < 3 && !rect; row++) for (let col = 0; col < 2; col++) {
           const candidate = {x: col ? .52 : .04, y: .04 + row * .32, w: .44, h: .27};
           if (canPlace(null, candidate)) { rect = candidate; break; }
         }
         if (!rect) { message.textContent = text('Free a slot before adding another panel.', 'Освободите место для новой панели.'); return; }
        const first = Object.keys(series)[0];
         const panel = {id: String(Date.now()) + Math.random(), series: first, color: defaultColor(first), customColor: false,
           rect};
         panels.push(panel); selected = panel.id; render();
         markCustom();
       }
       document.getElementById('addPlotPanel').onclick = addPanel;
       document.getElementById('removePanel').onclick = () => {
        const index = panels.findIndex(item => item.id === selected);
        if (index < 0) return;
         panels.splice(index, 1); selected = panels.at(-1)?.id || null; markCustom(); render();
      };
      select.onchange = () => { const panel = panels.find(item => item.id === selected); if (!panel) return;
          panel.series = select.value; panel.color = defaultColor(panel.series); panel.customColor = false;
          markCustom(); render(); };
       color.oninput = () => { const panel = panels.find(item => item.id === selected); if (!panel) return;
          panel.color = color.value; panel.customColor = true; markCustom(); render(); };
       epoch.onchange = render;
      function changeSize(axis, input) {
        const panel = panels.find(item => item.id === selected);
        if (!panel || !input.value) return;
         const candidate = {...panel.rect, [axis]: clamp(Number(input.value) / 100, .16,
           1 - panel.rect[axis === 'w' ? 'x' : 'y'])};
         if (!place(panel, candidate)) message.textContent = text('Panels cannot overlap.', 'Панели не могут перекрываться.');
         render();
      }
      width.onchange = () => changeSize('w', width);
      height.onchange = () => changeSize('h', height);
        const initialTemplate = Object.keys(templates).find(name => templates[name]);
        if (initialTemplate) applyTemplate(initialTemplate); else addPanel();
      document.getElementById('generatePlot').onclick = async () => {
         if (!panels.length) { message.textContent = text('Add at least one panel.', 'Добавьте хотя бы одну панель.'); return; }
         if (Object.keys(epochs).length && !epoch.value) {
           message.textContent = text('Selected maps have no common observation time.', 'У выбранных карт нет общего времени наблюдения.'); return;
         }
        const email = document.getElementById('plotEmail'); if (!email.reportValidity()) return;
        const button = document.getElementById('generatePlot'); button.disabled = true;
        message.textContent = text('Generating plot…', 'Построение графика…');
        try {
          const response = await fetch('/api/plots', {method: 'POST', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({event: eventPath, layout: 'free', style: style.value, epoch: epoch.value || null, title: document.getElementById('plotTitle').value,
               email: email.value, panels: panels.map(panel => ({series: panel.series, color: panel.color,
                 rect: panel.rect}))})});
          const result = await response.json();
          if (!response.ok) throw Error(result.error || 'Failed to create plot');
          document.getElementById('renderedImage').src = result.url;
          document.getElementById('renderedLink').href = result.url;
          document.getElementById('renderedPlot').hidden = false;
          message.textContent = text('Saved. Find it later by email in the catalog.', 'Сохранено. Позже найдите график по почте в каталоге.');
        } catch (error) { message.textContent = error.message; } finally { button.disabled = false; }
      };
    })();
    </script>
    """
    replacements = {"SERIES_JSON": data, "EPOCHS_JSON": times, "CHANNELS_JSON": channel_json,
                    "EVENT_JSON": event_path, "EVENT_NAME": name,
                    "HEADING": html.escape(heading), "SUBTITLE": html.escape(subtitle), "HEADING_JSON": heading_json,
                     "SUBTITLE_JSON": subtitle_json, "LOCATION_JSON": location_json,
                    "EVENT_URL": "/" + quote(event["path"], safe="/") + "/"}
    body = re.sub(r"__(SERIES_JSON|EPOCHS_JSON|CHANNELS_JSON|EVENT_JSON|EVENT_NAME|EVENT_URL|HEADING|SUBTITLE|HEADING_JSON|SUBTITLE_JSON|LOCATION_JSON)__",
                  lambda match: replacements[match.group(1)], body)
    css = """
    <style>
      main { width: min(1720px, calc(100vw - 36px)); padding-top: 22px; }
      .studio-heading { margin-bottom: 18px; }
      .studio-heading h1 { margin: 12px 0 4px; }
      .studio-heading p, .studio-back { color: var(--muted); }
      .studio-layout { display: grid; grid-template-columns: minmax(0, 1fr) 295px; gap: 16px; align-items: start; }
      .studio-workspace, .studio-sidebar { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); }
      .studio-toolbar { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; padding: 12px 16px; }
      .studio-toolbar span { flex: 1; color: var(--muted); font-size: 12px; }
      .studio-scroll { overflow-x: auto; padding: 14px; background: #edf2f7; border-radius: 0 0 12px 12px; }
       .plot-canvas { position: relative; width: 100%; min-width: 620px; aspect-ratio: 12 / 8.5; overflow: hidden; background: #fff;
         border: 1px solid #cbd5e1; box-shadow: 0 5px 22px #1422351c; }
       .plot-canvas[data-style="plotter"] { background: #f4f7fb; }
        .plot-canvas[data-style="simple"] .canvas-title { font-weight: 700; color: #111; }
       .plot-canvas[data-style="simple"] .canvas-subtitle { color: #333; }
       .plot-canvas[data-style="simple"] .canvas-panel { border-color: #ddd; box-shadow: none; }
       .plot-canvas[data-style="simple"] .canvas-panel-bar { color: #111; }
       .plot-canvas[data-style="simple"] .canvas-panel-bar strong { font-weight: 400; }
       .plot-canvas[data-style="simple"] .canvas-chart:not(.canvas-sun):not(.canvas-map) { border: 1px solid #111;
         background: repeating-linear-gradient(to bottom, transparent 0 24%, #e2e2e2 25% calc(25% + 1px)); }
       .plot-canvas[data-style="simple"] .canvas-chart:not(.canvas-sun):not(.canvas-map)::before {
         content: ''; position: absolute; inset: 0; pointer-events: none;
         background: repeating-linear-gradient(to right, transparent 0 24%, #e2e2e2 25% calc(25% + 1px)); }
       .plot-canvas[data-style="simple"] .canvas-xlabel, .plot-canvas[data-style="simple"] .canvas-ylabel,
       .plot-canvas[data-style="simple"] .canvas-ticks { color: #111; }
        .canvas-title { position: absolute; top: 1%; left: 5%; width: 90%; text-align: center; overflow: hidden;
         white-space: nowrap; text-overflow: ellipsis; font-size: clamp(12px, 1.4vw, 19px); }
        .canvas-subtitle { position: absolute; top: 5%; left: 5%; width: 90%; text-align: center; overflow: hidden;
          white-space: nowrap; text-overflow: ellipsis; font-size: 11px; color: #63738a; }
        .canvas-legend { position: absolute; top: 9%; left: 5%; width: 90%; text-align: center; overflow: hidden;
          white-space: nowrap; text-overflow: ellipsis; font-size: 10px; color: #63738a; }
      .plot-stage { position: absolute; left: 5.5%; top: 12.5%; width: 89%; height: 82%;
        background-image: linear-gradient(#eef2f7 1px, transparent 1px), linear-gradient(90deg, #eef2f7 1px, transparent 1px);
        background-size: 5% 5%; }
       .canvas-panel { position: absolute; min-width: 0; min-height: 0; background: white;
         border: 1px solid #dce4ee; box-shadow: 0 3px 12px #1e293b1c; overflow: hidden; color: #63738a; }
      .canvas-panel.active { border-color: var(--accent); box-shadow: 0 0 0 3px #2878a533; z-index: 2; }
       .canvas-panel-bar { position: absolute; left: 4%; top: 3%; width: 92%; height: 15%; display: flex; align-items: center;
         gap: 6px; cursor: grab; touch-action: none; user-select: none; color: #17243a; }
      .canvas-panel-bar:active { cursor: grabbing; }
       .canvas-panel-bar strong { position: absolute; left: 10%; width: 80%; text-align: center;
         overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }
       .canvas-panel-bar > span { position: relative; z-index: 1; }
          .canvas-chart { position: absolute; left: 15%; top: 17%; width: 79%; height: 64%;
         border-left: 1px solid #dce4ee; border-bottom: 1px solid #dce4ee;
         background: repeating-linear-gradient(to bottom, transparent 0 24%, #dce4ee 25% calc(25% + 1px)); }
        .canvas-chart.canvas-map { left: 15%; top: 18%; width: 73%; height: 62%; background: #edf2f7; }
         .canvas-panel[data-narrow="true"] .canvas-chart:not(.canvas-map):not(.canvas-sun) { left: 19%; width: 75%; }
        .canvas-panel:has(.canvas-map) .canvas-xlabel, .canvas-panel:has(.canvas-map) .canvas-ticks { left: 15%; width: 73%; }
       .canvas-chart svg { width: 100%; height: 100%; fill: currentColor; opacity: .7; }
       .canvas-map svg { color: #2a9d8f; }
       .canvas-chart.canvas-sun { left: 15%; top: 21%; width: 70%; height: 59%; border: 0; background: #101b2b; }
       .canvas-sun svg { opacity: 1; }
          .canvas-chart polyline { fill: none; stroke: currentColor; stroke-width: 2.5; vector-effect: non-scaling-stroke; }
          .canvas-chart polyline.canvas-secondary { stroke: #333333; stroke-dasharray: 5 4; }
         .canvas-second-line { position: absolute; right: 1%; bottom: 1%; color: #333; background: #ffffffe0;
           font-size: 8px; padding: 1px 3px; }
        .canvas-marker { position: absolute; top: 0; bottom: 0; z-index: 1; border-left: 1px dashed #7b8494; }
         .canvas-marker.peak { border-left: 1px dashed #bd4651; }
        .canvas-marker.map { border-left: 2px dotted #9a6400; }
         .canvas-xlabel { position: absolute; top: 85%; left: 15%; width: 79%; text-align: center; font-size: clamp(7px, .85vw, 11px); }
        .canvas-units { position: absolute; top: 13%; left: 5%; font-size: clamp(6px, .7vw, 9px); color: #63738a; }
       .canvas-ylabel { position: absolute; top: 42%; left: 1%; width: 17%; text-align: center; overflow-wrap: anywhere;
         font-size: clamp(6px, .75vw, 10px); }
         .canvas-ticks { position: absolute; top: 74%; left: 15%; width: 79%; text-align: center; white-space: pre; overflow: hidden;
          font-size: clamp(6px, .7vw, 9px); }
        .canvas-panel[data-narrow="true"]:not(:has(.canvas-map)):not(:has(.canvas-sun)) .canvas-xlabel,
        .canvas-panel[data-narrow="true"]:not(:has(.canvas-map)):not(:has(.canvas-sun)) .canvas-ticks { left: 19%; width: 75%; }
         .canvas-scale { position: absolute; top: 18%; left: 89%; width: 10%; height: 62%; display: flex; align-items: center;
         border-left: 6px solid #2a9d8f; font-size: clamp(6px, .6vw, 8px); overflow-wrap: anywhere; }
      .resize-handle { position: absolute; right: 0; bottom: 0; width: 20px; height: 20px; cursor: nwse-resize; touch-action: none;
        background: linear-gradient(135deg, transparent 49%, var(--accent) 50%, var(--accent) 57%, transparent 58%); }
      .studio-sidebar { padding: 18px; display: grid; gap: 12px; }
      .studio-sidebar h2, .studio-sidebar h3 { margin: 0; font-size: 17px; }
      .studio-sidebar h3 { font-size: 14px; }
      .studio-sidebar label, .panel-settings { display: grid; gap: 5px; }
      .studio-sidebar input, .studio-sidebar select { width: 100%; min-height: 38px; padding: 6px; border: 1px solid var(--line); border-radius: 7px; font: inherit; }
      .studio-sidebar input[type=color] { padding: 3px; }
      .studio-note { color: var(--muted); font-size: 12px; margin: 0; }
      .panel-settings { gap: 12px; border-top: 1px solid var(--line); padding-top: 16px; }
        .panel-settings[hidden], #colorField[hidden], #mapPalette[hidden], #renderedPlot[hidden] { display: none; }
      .size-fields { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
      .studio-generate { background: var(--accent); color: white; cursor: pointer; }
      .studio-generate:disabled { opacity: .55; cursor: wait; }
      #plotMessage { overflow-wrap: anywhere; }
      #renderedPlot img { display: block; width: 100%; margin-top: 8px; border: 1px solid var(--line); }
      @media (max-width: 950px) { .studio-layout { grid-template-columns: 1fr; } .studio-sidebar { grid-template-columns: repeat(2, minmax(0, 1fr)); } .studio-sidebar h2, .panel-settings, .studio-note, #plotMessage, #renderedPlot { grid-column: 1 / -1; } }
      @media (max-width: 560px) { .studio-sidebar { grid-template-columns: 1fr; } .studio-sidebar > * { grid-column: 1 / -1; } }
    </style>
    """
    return body, css
