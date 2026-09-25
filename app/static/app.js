const $ = (id) => document.getElementById(id);

async function refreshStatus() {
  const response = await fetch('/api/status');
  const data = await response.json();
  $('status-dot').classList.toggle('ready', data.ready);
  $('status-text').textContent = data.ready ? 'Model ready' : 'Train model to activate';
  $('analyzed').textContent = data.analyzed;
  $('spam-count').textContent = data.spam_count;
  $('legit-count').textContent = data.legitimate_count;
  $('f1').textContent = data.metrics ? `${(data.metrics.f1 * 100).toFixed(1)}%` : '--';
  const rows = data.recent || [];
  $('history').innerHTML = rows.length ? rows.map((item, index) => `<div class="history-row"><span>Analysis ${rows.length - index}</span><b class="${item.prediction === 'SPAM' ? 'spam' : 'legit'}">${item.prediction}</b><span>${(item.confidence * 100).toFixed(1)}% confidence</span></div>`).join('') : 'No classifications in this session yet.';
}

$('analyzer-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('error').textContent = '';
  $('analyze').disabled = true;
  $('analyze').firstChild.textContent = 'Analyzing... ';
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: new FormData(event.target) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed.');
    const spam = data.prediction === 'SPAM';
    $('verdict').className = `verdict ${spam ? 'spam' : 'legit'}`;
    $('verdict').innerHTML = `<div class="verdict-icon">${spam ? '!' : '✓'}</div><h3>${data.prediction}</h3><p>${spam ? 'Treat this message with caution.' : 'No strong spam pattern detected.'}</p>`;
    $('result-number').textContent = spam ? '01' : '00';
    $('probability').textContent = `${(data.probability_spam * 100).toFixed(1)}%`;
    $('meter-fill').style.width = `${data.probability_spam * 100}%`;
    $('risk-chip').textContent = `RISK ${data.risk}`;
    $('confidence-chip').textContent = `CONFIDENCE ${(data.confidence * 100).toFixed(1)}%`;
    $('explanations').innerHTML = data.explanations.map(item => `<li>${item.message}</li>`).join('');
    await refreshStatus();
  } catch (error) { $('error').textContent = error.message; }
  finally { $('analyze').disabled = false; $('analyze').firstChild.textContent = 'Analyze email '; }
});

refreshStatus().catch(() => { $('status-text').textContent = 'Service unavailable'; });
