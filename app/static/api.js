async function readJSON(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(`The server returned an unreadable response (${response.status}).`);
  }
  if (!response.ok) {
    throw new Error(payload && payload.error ? payload.error : `Request failed (${response.status}).`);
  }
  return payload;
}

export async function getJSON(path) {
  return readJSON(await fetch(path, { cache: 'no-store', headers: { Accept: 'application/json' } }));
}

export async function postForm(path, formData, signal) {
  return readJSON(await fetch(path, {
    method: 'POST', body: formData, cache: 'no-store', signal,
    headers: { 'Cache-Control': 'no-store', Pragma: 'no-cache' }
  }));
}

export async function analyse({ subject, body, file, runRobustness, runComparisons, runAttribution, signal }) {
  const form = new FormData();
  form.set('subject', subject || '');
  form.set('body', body || '');
  form.set('run_robustness', runRobustness ? 'true' : 'false');
  form.set('run_comparisons', runComparisons ? 'true' : 'false');
  form.set('run_attribution', runAttribution ? 'true' : 'false');
  if (file) form.set('email_file', file, file.name);
  return postForm('/api/analyze', form, signal);
}

export const endpoints = {
  status: () => getJSON('/api/status'),
  probes: () => getJSON('/api/probes'),
  health: () => getJSON('/api/health'),
  calibration: () => getJSON('/api/calibration'),
  evaluation: () => getJSON('/api/evaluation'),
  history: () => getJSON('/api/history'),
  clearHistory: () => fetch('/api/history/clear', { method: 'POST' }).then(readJSON)
};
