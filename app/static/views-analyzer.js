import {
  bullet, clear, el, integer, kvGrid, mount, ms, pct, setIcon, setText, show, table, tag
} from './dom.js';

export function renderStats(status) {
  const container = document.getElementById('stat-row');
  if (!container) return;
  const historyData = status.history || {};
  const entries = [
    ['EMAILS CHECKED', integer(historyData.analyzed), ''],
    ['SPAM', integer(historyData.spam_count), 'danger'],
    ['NOT SPAM', integer(historyData.legitimate_count), 'safe']
  ];
  clear(container);
  entries.forEach(([label, value, variant]) => {
    container.appendChild(
      el('div', { class: 'stat' }, [
        el('span', { text: label }),
        el('strong', { class: variant, text: value })
      ])
    );
  });
}

export function renderVerdict(result) {
  const model = result.model || {};
  const isSpam = model.prediction === 'SPAM';
  show('verdict-empty', false);
  show('verdict-body', true);

  const panel = document.getElementById('verdict');
  panel.className = `verdict ${isSpam ? 'is-spam' : 'is-legit'}`;
  setText('verdict-label', isSpam ? 'SPAM' : 'NOT SPAM');
  setIcon('verdict-icon', isSpam ? 'alert' : 'check');
  setText(
    'verdict-note',
    isSpam
      ? 'Be careful with this message. Check the sender before responding or opening links.'
      : 'No strong warning signs were found, but you should still use your usual caution.'
  );

  const chip = document.getElementById('verdict-chip');
  chip.className = `chip ${isSpam ? 'chip-danger' : 'chip-safe'}`;
  setText('verdict-chip', `RISK ${model.risk || '--'}`);

  const rawProbability = Number(model.probability_spam);
  const calibratedProbability = model.calibrated_confidence;
  const evidenceAdjustedProbability = model.evidence_adjusted_probability;
  const hasCalibratedProbability =
    typeof calibratedProbability === 'number' && Number.isFinite(calibratedProbability);
  const hasEvidenceAdjustedProbability =
    typeof evidenceAdjustedProbability === 'number' && Number.isFinite(evidenceAdjustedProbability);
  const displayProbability = hasEvidenceAdjustedProbability
    ? evidenceAdjustedProbability
    : (hasCalibratedProbability ? calibratedProbability : rawProbability);
  setText('probability-value', pct(displayProbability, 2));
  const meter = document.getElementById('meter-fill');
  meter.style.width = `${Math.min(100, Math.max(0, displayProbability * 100))}%`;
  meter.parentElement.className = `meter ${isSpam ? 'is-spam' : 'is-legit'}`;

  const note = document.getElementById('calibration-note');
  if (note) {
    if (!hasCalibratedProbability) {
      note.hidden = true;
      note.textContent = '';
    } else {
      const distance = Math.abs(calibratedProbability - rawProbability);
      const direction = calibratedProbability > rawProbability ? 'higher' : 'lower';
      note.hidden = false;
      note.textContent =
        `Calibrated on a held-out validation set. Raw model probability: ${pct(rawProbability, 2)}. ` +
        `${distance < 0.005 ? 'Calibration is effectively unchanged.' : `Calibration is ${direction} by ${(distance * 100).toFixed(2)} percentage points.`} ` +
        `${model.evidence
          ? `The displayed chance is conservatively moved toward 50% (evidence weight ${((model.evidence.display_weight === undefined ? 1 : model.evidence.display_weight) * 100).toFixed(0)}%; ${model.evidence.token_count} tokens; ${(model.evidence.oov_rate * 100).toFixed(0)}% unfamiliar). `
          : ''}The spam/not-spam decision is unchanged by temperature scaling at the 0.50 threshold.`;
    }
  }
}

export function renderPersuasionReport(signals) {
  const container = document.getElementById('persuasion-list');
  const list = signals.persuasion || [];
  if (!list.length) {
    mount(container, bullet('No persuasion-signal patterns matched this email.', 'check'));
    return;
  }
  mount(container, list.map((signal) => el('div', { class: 'signal' }, [
    el('div', { class: 'signal-head' }, [
      el('span', { text: signal.label }),
      el('span', { class: 'count', text: `${signal.count}x` })
    ]),
    el('p', { text: (signal.evidence || []).join('  /  ') })
  ])));
}

export function renderSecuritySignals(signals) {
  const structure = signals.structure || {};
  mount(document.getElementById('structure-grid'), kvGrid([
    ['CHARACTERS', integer(structure.characters)],
    ['WORDS', integer(structure.words)],
    ['UPPERCASE RATIO', structure.uppercase_ratio === undefined ? 'Not available' : pct(structure.uppercase_ratio)],
    ['EXCLAMATION MARKS', integer(structure.exclamation_count)],
    ['CURRENCY SYMBOLS', integer(structure.currency_symbol_count)],
    ['SHOUTING WORDS', integer(structure.shouting_words)],
    ['CONTAINS HTML', structure.contains_html ? 'yes' : 'no'],
    ['REMOTE IMAGES', integer(structure.remote_image_count)],
    ['MENTIONS ATTACHMENT', structure.mentions_attachment ? 'yes' : 'no'],
    ['URL MASKED FOR MODEL', structure.model_saw_url_placeholder ? 'yes' : 'no']
  ]));

  const findings = signals.url_findings || [];
  mount(document.getElementById('url-findings'),
    findings.length
      ? findings.map((finding) => bullet(finding, 'link'))
      : [bullet('No URL-level findings were triggered.', 'check')]);

  const hidden = signals.hidden_content || [];
  if (hidden.length) {
    document.getElementById('url-findings').appendChild(
      bullet(
        `Hidden-content techniques found in the HTML source: ${hidden
          .map((item) => `${item.technique} (${item.occurrences})`)
          .join(', ')}.`,
        'eyeOff'
      )
    );
  }

  const urls = signals.urls || [];
  mount(document.getElementById('url-table'), table(
    ['DEFANGED URL', 'HOST', 'SCHEME', 'PATH DEPTH', 'FLAGS'],
    urls.slice(0, 40).map((url) => [
      { text: url.url, mono: true },
      { text: url.host || '--', mono: true },
      url.scheme || '--',
      String(url.path_depth),
      (url.flags || []).map((flag) => tag(
        flag,
        flag === 'plain_http' || flag === 'ip_literal_host' ? 'danger' : 'amber'
      ))
    ]),
    { empty: 'No URLs were found in this email.' }
  ));
}
export function renderUnicode(unicode) {
  const findings = unicode.findings || [];
  const mixed = unicode.mixed_script_tokens || [];
  mount(document.getElementById('unicode-summary'), kvGrid([
    ['DISTINCT FINDINGS', integer(unicode.total_findings)],
    ['INVISIBLE CHARACTERS', integer(unicode.invisible_count)],
    ['MIXED-SCRIPT TOKENS', integer(mixed.length)],
    ['VERDICT', findings.length ? 'characters worth reviewing' : 'nothing unusual found',
      findings.length ? 'amber' : 'safe']
  ]));

  mount(document.getElementById('unicode-table'), table(
    ['KIND', 'CHARACTER', 'CODE POINT', 'UNICODE NAME', 'CATEGORY', 'COUNT', 'REMOVED BY HARDENING', 'CONTEXT'],
    findings.map((item) => [
      tag(item.kind.replace(/_/g, ' '), item.kind === 'zero_width' || item.kind === 'bidi_control' ? 'danger' : 'amber'),
      { text: item.character === '\u200b' ? 'U+200B' : `'${item.character}'`, mono: true },
      { text: item.codepoint, mono: true },
      item.name,
      item.category,
      String(item.count),
      item.removed_by_hardening ? tag('removed', 'safe') : tag('kept', 'amber'),
      { text: (item.contexts || []).join('  |  '), mono: true }
    ]),
    { empty: 'No invisible, format, control or exotic-space characters were detected.' }
  ));

  if (mixed.length) {
    document.getElementById('unicode-table').appendChild(
      table(
        ['MIXED-SCRIPT TOKEN', 'SCRIPTS', 'LATIN FOLDED FORM', 'OCCURRENCES'],
        mixed.map((token) => [
          { text: token.token, mono: true },
          (token.scripts || []).join(' + '),
          { text: token.folded, mono: true },
          String(token.count)
        ])
      )
    );
  }
}

export function renderRepresentations(normalization) {
  const comparison = normalization.comparison || {};
  const differences = normalization.differences || {};
  const views = normalization.views || [];
  const asShipped = comparison.shipped || {};
  const hardened = comparison.hardened || {};
  const raw = comparison.raw || {};

  const chip = document.getElementById('normalization-chip');
  const identical = differences.shipped_equals_hardened;
  chip.textContent = identical ? 'NORMALIZATION NO-OP' : 'NORMALIZATION CHANGED TEXT';
  chip.className = `chip ${identical ? 'chip-safe' : 'chip-warn'}`;

  mount(document.getElementById('representation-table'), table(
    ['REPRESENTATION', 'PURPOSE', 'PREDICTION', 'PROBABILITY', 'CHANGE vs AS-SHIPPED'],
    [
      ['Raw input', 'No normalisation applied', raw.label || '--', pct(raw.probability, 2),
        { text: deltaText(comparison.raw_vs_shipped_delta), class: 'num' }],
      ['As-shipped pipeline', 'The production decision', asShipped.label || '--', pct(asShipped.probability, 2),
        { text: 'baseline', class: 'num delta-zero' }],
      ['Security-hardened', 'Invisible characters removed, homoglyphs folded',
        hardened.label || '--', pct(hardened.probability, 2),
        { text: deltaText(comparison.hardened_vs_shipped_delta), class: 'num' }]
    ]
  ));

  const viewByKey = {};
  views.forEach((view) => { viewByKey[view.key] = view; });
  const shippedText = (viewByKey.shipped || {}).combined_text || '';
  const hardenedText = (viewByKey.hardened || {}).combined_text || '';
  document.getElementById('shipped-text').textContent = shippedText || 'Not available';
  document.getElementById('hardened-text').textContent = hardenedText || 'Not available';

  const tokenChanges = differences.token_changes || [];
  mount(document.getElementById('token-diff'), table(
    ['AS-SHIPPED TOKEN', 'HARDENED TOKEN'],
    tokenChanges.map((change) => [
      { text: orPlaceholder(change.before), mono: true },
      { text: orPlaceholder(change.after), mono: true }
    ]),
    {
      empty: identical
        ? 'Hardening produced an identical token stream: this email contained nothing to normalise.'
        : 'Token boundaries match; only character-level content differs.'
    }
  ));

  if (!identical && differences.removed_character_kinds) {
    const kinds = Object.entries(differences.removed_character_kinds || {})
      .filter(([, count]) => count > 0)
      .map(([kind, count]) => `${kind}: ${count}`);
    if (kinds.length) {
      document.getElementById('token-diff').appendChild(
        bullet(`Characters eliminated by hardening - ${kinds.join(', ')}.`, 'eyeOff')
      );
    }
  }
}

function deltaText(value) {
  if (value === null || value === undefined) return 'Not available';
  const sign = value > 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(2)} pp`;
}

function orPlaceholder(value) {
  return value === '' || value === null || value === undefined ? '(none)' : value;
}

export function renderAttribution(attribution) {
  if (!attribution || attribution.available === false) {
    mount(document.getElementById('attribution-summary'), kvGrid([
      ['METHOD', 'position-preserving occlusion'],
      ['STATUS', (attribution && attribution.reason) || 'Not available', 'muted']
    ]));
    mount(document.getElementById('attribution-table'), table(
      ['TOKEN', 'OCCURRENCES', 'OCCLUDED PROBABILITY', 'DELTA', 'DIRECTION'],
      [],
      { empty: 'No attribution was computed for this analysis.' }
    ));
    return;
  }

  mount(document.getElementById('attribution-summary'), kvGrid([
    ['METHOD', attribution.method || 'Not available'],
    ['BASELINE', pct(attribution.baseline_probability, 2)],
    ['TOKENS EVALUATED', integer(attribution.evaluated_tokens)],
    ['UNIQUE TOKENS', integer(attribution.unique_tokens)],
    ['NEGLIGIBLE BAND', `|delta| < ${attribution.negligible_threshold}`],
    ['TRUNCATED', attribution.truncated ? 'yes' : 'no']
  ]));

  const tokens = attribution.tokens || [];
  mount(document.getElementById('attribution-table'), table(
    ['TOKEN', 'OCCURRENCES', 'OCCLUDED PROBABILITY', 'DELTA (BASELINE - OCCLUDED)', 'DIRECTION'],
    tokens.map((token) => [
      { text: token.token, mono: true },
      String(token.occurrences),
      { text: pct(token.occluded_probability, 2), class: 'num' },
      { text: `${token.delta > 0 ? '+' : ''}${token.delta.toFixed(4)}`, class: `num ${token.delta > 0 ? 'delta-pos' : token.delta < 0 ? 'delta-neg' : 'delta-zero'}` },
      tag(
        token.direction.replace(/_/g, ' '),
        token.direction === 'supports_spam' ? 'danger' : token.direction === 'suppresses_spam' ? 'safe' : ''
      )
    ]),
    { empty: 'No tokens were available for attribution.' }
  ));
}
