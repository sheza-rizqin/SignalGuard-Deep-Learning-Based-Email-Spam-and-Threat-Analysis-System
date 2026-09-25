import { analyse, endpoints } from './api.js';
import { applyStaticIcons, setText } from './dom.js';
import {
  renderStats,
  renderVerdict
} from './views-analyzer.js';
import {
  renderArtifacts,
  renderEvasionLab,
  renderHealth,
  renderHistory
} from './views-labs.js';

function setStatus(ready) {
  document.getElementById('status-dot').classList.toggle('ready', ready);
  setText('status-text', ready ? 'Model ready' : 'Train model to activate');
}

async function refreshStatus() {
  const status = await endpoints.status();
  setStatus(status.ready);
  renderStats(status);
  renderArtifacts(status.artifacts);
  renderHistory(await endpoints.history());
}

async function loadLabData() {
  const results = await Promise.allSettled([
    endpoints.health()
  ]);
  const [health] = results;
  if (health.status === 'fulfilled') renderHealth(health.value);
}

function bindTabs() {
  document.querySelectorAll('[data-panel]').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((item) => item.classList.toggle('is-active', item === tab));
      document.querySelectorAll('.panel').forEach((panel) => {
        panel.classList.toggle('is-visible', panel.id === `panel-${tab.dataset.panel}`);
      });
    });
  });
}

function loadEmailFromUrl() {
  const parameters = new URLSearchParams(window.location.search);
  const subject = parameters.get('subject');
  const body = parameters.get('body');
  if (subject === null && body === null) return;

  document.getElementById('subject').value = subject || '';
  document.getElementById('body').value = body || '';
  setText('file-status', 'Example email loaded from the link. Review it, then select Check email.');
  window.history.replaceState({}, document.title, window.location.pathname);
}

function bindAnalyzer() {
  const form = document.getElementById('analyzer-form');
  const fileInput = document.getElementById('email-file');
  const button = document.getElementById('analyze-button');
  const subjectInput = document.getElementById('subject');
  const bodyInput = document.getElementById('body');
  let activeRequest = null;

  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) return;
    setText('file-status', `${file.name} ready to analyse`);
    setText('error', '');
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    setText('error', '');
    button.disabled = true;
    setText('analyze-label', 'Analyzing...');
    if (activeRequest) activeRequest.abort();
    const controller = new AbortController();
    activeRequest = controller;
    const selectedFile = fileInput.files[0];
    let uploadCompleted = false;
    try {
      const result = await analyse({
        subject: subjectInput.value,
        body: bodyInput.value,
        file: selectedFile,
        runRobustness: document.getElementById('opt-robustness').checked,
        runComparisons: false,
        runAttribution: false,
        signal: controller.signal
      });
      if (selectedFile && result.content) {
        subjectInput.value = result.content.subject || '';
        bodyInput.value = result.content.body || '';
        setText('file-status', `${selectedFile.name} analysed and loaded into the fields`);
        uploadCompleted = true;
      }
      renderVerdict(result);
      renderEvasionLab(result.robustness);
      await refreshStatus();
      await loadLabData();
    } catch (error) {
      if (error.name !== 'AbortError') setText('error', error.message);
    } finally {
      if (uploadCompleted && fileInput.files.length) {
        fileInput.value = '';
      } else if (selectedFile && !uploadCompleted && fileInput.files.length) {
        setText('file-status', `${selectedFile.name} is still selected — fix the error and try again`);
      }
      button.disabled = false;
      setText('analyze-label', 'Check email');
      if (activeRequest === controller) activeRequest = null;
    }
  });
}

function bindHistory() {
  document.getElementById('clear-history').addEventListener('click', async () => {
    await endpoints.clearHistory();
    renderHistory(await endpoints.history());
  });
}

async function start() {
  applyStaticIcons();
  loadEmailFromUrl();
  bindTabs();
  bindAnalyzer();
  bindHistory();
  try {
    await refreshStatus();
    await loadLabData();
  } catch (error) {
    setStatus(false);
    setText('error', error.message);
  }
}

start();
