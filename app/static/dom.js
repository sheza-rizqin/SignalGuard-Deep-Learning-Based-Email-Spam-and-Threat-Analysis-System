const ICON_PATHS = {
  scan: '<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/>',
  lab: '<path d="M9 3h6M10 3v6l-5 8a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-8V3"/><path d="M8 15h8"/>',
  gauge: '<path d="M12 21a9 9 0 1 1 9-9"/><path d="M12 12l5-3"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  bolt: '<path d="M13 3 4 14h6l-1 7 9-11h-6l1-7z"/>',
  pulse: '<path d="M3 12h4l2-6 3 12 3-9 2 3h4"/>',
  list: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2"/>',
  alert: '<path d="M12 3 2 20h20L12 3z"/><path d="M12 10v4M12 17h.01"/>',
  check: '<path d="M4 12.5 9 17.5 20 6.5"/>',
  eyeOff: '<path d="M3 3l18 18"/><path d="M10.6 6.2A9.5 9.5 0 0 1 12 6c5 0 9 6 9 6a17 17 0 0 1-3.4 3.9M6.3 7.5A17 17 0 0 0 3 12s4 6 9 6a9.4 9.4 0 0 0 3.4-.6"/>',
  link: '<path d="M10 13a5 5 0 0 0 7 0l2-2a5 5 0 0 0-7-7l-1 1"/><path d="M14 11a5 5 0 0 0-7 0l-2 2a5 5 0 0 0 7 7l1-1"/>',
  shield: '<path d="M12 3l7 3v6c0 4.3-2.9 7.7-7 9-4.1-1.3-7-4.7-7-9V6l7-3z"/>',
  dot: '<circle cx="12" cy="12" r="4"/>'
};

export function icon(name, extraClass = '') {
  const span = document.createElement('span');
  span.className = extraClass ? `ico ${extraClass}` : 'ico';
  span.setAttribute('aria-hidden', 'true');
  span.innerHTML =
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" ` +
    `stroke-linecap="round" stroke-linejoin="round">${ICON_PATHS[name] || ICON_PATHS.dot}</svg>`;
  return span;
}

export function applyStaticIcons(root = document) {
  root.querySelectorAll('[data-icon]').forEach((node) => {
    const name = node.getAttribute('data-icon');
    node.innerHTML =
      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" ` +
      `stroke-linecap="round" stroke-linejoin="round">${ICON_PATHS[name] || ICON_PATHS.dot}</svg>`;
  });
}

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (value === undefined || value === null || value === false) return;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = String(value);
    else if (key === 'html') node.innerHTML = value;
    else node.setAttribute(key, value);
  });
  const list = Array.isArray(children) ? children : [children];
  list.forEach((child) => {
    if (child === null || child === undefined || child === false) return;
    node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  });
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

export function mount(node, ...children) {
  clear(node);
  children.flat().forEach((child) => {
    if (child === null || child === undefined || child === false) return;
    node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  });
  return node;
}

export function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

export function isAvailable(value) {
  return value !== null && value !== undefined;
}

export function pct(value, digits = 1) {
  if (!isAvailable(value) || Number.isNaN(value)) return 'Not available';
  return `${(value * 100).toFixed(digits)}%`;
}

export function fixed(value, digits = 4) {
  if (!isAvailable(value) || Number.isNaN(value)) return 'Not available';
  return Number(value).toFixed(digits);
}

export function ms(value) {
  if (!isAvailable(value) || Number.isNaN(value)) return 'Not available';
  return `${Number(value).toFixed(1)} ms`;
}

export function bytes(value) {
  if (!isAvailable(value) || !value) return 'Not available';
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = Number(value);
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function integer(value) {
  if (!isAvailable(value) || Number.isNaN(value)) return 'Not available';
  return Number(value).toLocaleString('en-US');
}

export function deltaClass(value) {
  if (!isAvailable(value) || Math.abs(value) < 1e-9) return 'delta-zero';
  return value > 0 ? 'delta-pos' : 'delta-neg';
}

export function deltaLabel(value, digits = 4) {
  if (!isAvailable(value) || Number.isNaN(value)) return 'Not available';
  const sign = value > 0 ? '+' : '';
  return `${sign}${Number(value).toFixed(digits)}`;
}
export function kvGrid(entries) {
  const grid = el('dl', { class: 'kv-grid' });
  entries.forEach(([label, value, className]) => {
    grid.appendChild(
      el('div', { class: 'kv' }, [
        el('dt', { text: label }),
        el('dd', {
          class: className || '',
          text: value === null || value === undefined ? 'Not available' : String(value)
        })
      ])
    );
  });
  return grid;
}

function toCell(value, headerIndex) {
  if (value && value.nodeType === 1) return el('td', {}, [value]);
  if (value && typeof value === 'object' && 'text' in value) {
    const classes = [value.class || '', value.num ? 'num' : '', value.mono ? 'mono' : ''];
    return el('td', { class: classes.join(' ').trim(), text: value.text });
  }
  return el('td', {
    class: headerIndex > 0 ? 'num' : '',
    text: value === null || value === undefined ? 'Not available' : String(value)
  });
}

export function table(headers, rows, options = {}) {
  const wrap = el('div', { class: 'table-wrap' });
  const node = el('table');
  const head = el('thead');
  head.appendChild(el('tr', {}, headers.map((header) => el('th', { text: String(header) }))));
  node.appendChild(head);
  const body = el('tbody');
  if (!rows.length) {
    body.appendChild(
      el('tr', {}, [
        el('td', { colspan: String(headers.length), class: 'mono', text: options.empty || 'No rows to display.' })
      ])
    );
  } else {
    rows.forEach((row) => {
      body.appendChild(el('tr', {}, row.map((cell, index) => toCell(cell, index))));
    });
  }
  node.appendChild(body);
  wrap.appendChild(node);
  return wrap;
}

export function tag(text, variant = '') {
  return el('span', { class: variant ? `tag ${variant}` : 'tag', text });
}

export function bullet(text, iconName = 'dot') {
  return el('div', { class: 'bullet' }, [icon(iconName), el('span', { text })]);
}

export function show(nodeId, visible) {
  const node = document.getElementById(nodeId);
  if (node) node.hidden = !visible;
}
export function setIcon(target, name) {
  const node = typeof target === 'string' ? document.getElementById(target) : target;
  if (!node) return;
  node.setAttribute('data-icon', name);
  node.innerHTML =
    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" ` +
    `stroke-linecap="round" stroke-linejoin="round">${ICON_PATHS[name] || ICON_PATHS.dot}</svg>`;
}
