import { bullet, el, integer, kvGrid, mount, ms, pct, show, table, tag } from './dom.js';

const FRIENDLY_CHECK_NAMES = {
  'Intra-word dot': 'Dots added inside words',
  'Leetspeak substitution': 'Letters changed to numbers',
  'Cyrillic homoglyphs': 'Look-alike letters',
  'Zero-width space': 'Hidden spaces',
  'Soft hyphen': 'Invisible word breaks',
  'Word joiner': 'Hidden word joiners',
  'Bidirectional marks': 'Hidden reading-direction marks',
  'Punctuation removal': 'Punctuation removed',
  'Punctuation collapse': 'Repeated punctuation simplified',
  'Punctuation doubling': 'Extra punctuation added',
  'Full-width characters': 'Wide characters',
  'Mathematical alphanumerics': 'Decorative characters',
  'Neutral word padding': 'Extra harmless words',
  'URL uppercasing': 'Link capitalization changed',
  'URL angle brackets': 'Link brackets changed',
  'URL defanging': 'Link punctuation changed',
  'Benign URL query': 'A harmless link detail added',
  'Uppercase': 'All capitals',
  'Lowercase': 'All lowercase',
  'Title case': 'Title-style capitals',
  'Alternating case': 'Mixed capitalization',
  'Space padding': 'Extra spaces',
  'Tab separators': 'Tabs between words',
  'Newline injection': 'New lines between words'
};

function friendlyCheckName(name) {
  return FRIENDLY_CHECK_NAMES[name] || name;
}

export function renderEvasionLab(robustness) {
  const available = robustness && robustness.available !== false;
  show('evasion-empty', !available);
  show('evasion-body', available);
  if (!available) return;

  const damaging = robustness.most_damaging;
  mount(document.getElementById('evasion-summary'), kvGrid([
    ['STARTING RESULT', robustness.baseline_label],
    ['SAME ANSWER EACH TIME', pct(robustness.prediction_stability, 1),
      robustness.prediction_stability >= 0.95 ? 'safe' : robustness.prediction_stability >= 0.8 ? 'amber' : 'danger'],
    ['ANSWERS THAT CHANGED', integer(robustness.changed), robustness.changed ? 'danger' : 'safe'],
    ['BIGGEST EFFECT', damaging ? friendlyCheckName(damaging.transformation.name) : 'none',
      damaging && damaging.prediction_changed ? 'danger' : 'amber'],
    ['CHECKS WITH THE SAME ANSWER', `${robustness.preserved} of ${robustness.evaluated}`],
    ['STOPPED BY NORMALIZATION', `${robustness.normalization_neutralised} of ${robustness.evaluated}`, 'safe'],
    ['SAME MODEL TOKENS', `${robustness.tokenizer_neutralised} of ${robustness.evaluated}`, 'safe'],
    ['REACHED A DIFFERENT MODEL INPUT', `${robustness.reached_model} of ${robustness.evaluated}`]
  ]));

  const variants = robustness.variants || [];
  const stageOf = (variant) => variant.pipeline_stage || (
    variant.normalization_neutralised ? 'normalization_neutralised'
      : variant.tokenizer_neutralised ? 'tokenizer_neutralised'
        : 'model_input_changed'
  );
  mount(document.getElementById('evasion-table'), table(
    ['VARIATION APPLIED TO THIS EMAIL', 'PIPELINE RESULT', 'BASELINE CHANCE', 'TESTED CHANCE', 'MOVEMENT', 'VERDICT CHANGED?'],
    variants.map((variant) => {
      const info = variant.transformation || {};
      const flipped = variant.prediction_changed;
      const stage = stageOf(variant);
      const pipelineResult = stage === 'normalization_neutralised'
        ? tag('removed by normalizer', 'safe')
        : stage === 'tokenizer_neutralised'
          ? tag('same model tokens', 'amber')
          : tag('new model input', 'danger');
      return [
        { text: friendlyCheckName(info.name), mono: false },
        pipelineResult,
        { text: pct(robustness.baseline_probability, 1), class: 'num' },
        { text: pct(variant.probability, 1), class: 'num' },
        { text: `${variant.probability_delta > 0 ? '+' : ''}${(variant.probability_delta * 100).toFixed(1)} pp`, class: 'num' },
        flipped ? tag('YES', 'danger') : tag('NO', 'safe')
      ];
    }),
    {
      empty: 'No safety variations were evaluated.'
    }
  ));
}

export function renderProbeCatalog(catalog) {
  const count = document.getElementById('probe-count');
  if (count) count.textContent = `${catalog.count} PROBES / ${(catalog.families || []).length} FAMILIES`;
  mount(document.getElementById('probe-table'), table(
    ['PROBE', 'FAMILY', 'WHAT IT DOES', 'WHY IT PRESERVES MEANING', 'STATED HYPOTHESIS'],
    (catalog.transformations || []).map((probe) => [
      { text: probe.name, mono: false },
      probe.family.replace(/_/g, ' '),
      probe.description,
      probe.rationale,
      probe.hypothesis
    ])
  ));
}

export function renderRobustnessArtifact(artifact) {
  const available = artifact && artifact.available === true;
  show('robustness-empty', !available);
  show('robustness-body', available);
  const chip = document.getElementById('robustness-chip');
  chip.textContent = available ? 'GENERATED' : 'NOT GENERATED';
  chip.className = `chip ${available ? 'chip-artifact' : 'chip-danger'}`;
  if (!available) {
    if (artifact && artifact.reason) document.getElementById('robustness-note').textContent = artifact.reason;
    return;
  }

  document.getElementById('robustness-note').textContent = artifact.definition || '';

  mount(document.getElementById('robustness-summary'), kvGrid([
    ['CORPUS SIZE', integer(artifact.corpus_size)],
    ['BASELINE SPAM RATE', pct(artifact.baseline_spam_rate, 1)],
    ['CORPUS STABILITY', pct(artifact.corpus_stability, 1),
      artifact.corpus_stability >= 0.95 ? 'safe' : artifact.corpus_stability >= 0.8 ? 'amber' : 'danger'],
    ['PROBES', integer((artifact.per_probe || []).length)]
  ]));

  mount(document.getElementById('robustness-family-table'), table(
    ['FAMILY', 'PROBES', 'MEAN STABILITY'],
    (artifact.by_family || []).map((family) => [
      family.family.replace(/_/g, ' '),
      String(family.probes),
      { text: pct(family.stability, 2), class: 'num' }
    ])
  ));

  mount(document.getElementById('robustness-table'), table(
    ['PROBE', 'FAMILY', 'EVALUATED', 'DECISIONS FLIPPED', 'STABILITY', 'MEAN |SHIFT|'],
    (artifact.per_probe || []).map((probe) => [
      probe.name,
      probe.family.replace(/_/g, ' '),
      String(probe.evaluated),
      String(probe.flips),
      { text: pct(probe.stability, 2), class: 'num' },
      { text: probe.mean_absolute_shift.toFixed(4), class: 'num' }
    ])
  ));
}
export function renderDrift(artifact) {
  const available = artifact && artifact.available === true;
  show('drift-empty', !available);
  const body = document.getElementById('drift-body');
  body.hidden = !available;
  const chip = document.getElementById('drift-chip');
  chip.textContent = available ? 'GENERATED' : 'NOT GENERATED';
  chip.className = `chip ${available ? 'chip-artifact' : 'chip-danger'}`;
  if (!available) {
    if (artifact && artifact.reason) {
      document.getElementById('drift-empty').textContent = artifact.reason;
    }
    return;
  }

  const container = document.getElementById('drift-body');
  mount(container);

  const corpus = artifact.corpus || {};
  container.appendChild(kvGrid([
    ['GENERATED', artifact.generated_at || 'Not available'],
    ['POOLED ROWS', integer(corpus.rows)],
    ['SOURCES', String((corpus.sources || []).length)],
    ['TIMESTAMPS', artifact.timestamps_available ? 'available' : 'not available',
      artifact.timestamps_available ? 'safe' : 'amber']
  ]));

  container.appendChild(table(
    ['SOURCE', 'ROWS', 'SPAM', 'HAM', 'SPAM RATE', 'FIRST', 'LAST'],
    (corpus.sources || []).map((source) => [
      source.source,
      String(source.rows),
      String(source.spam),
      String(source.ham),
      { text: pct(source.spam / Math.max(source.rows, 1), 1), class: 'num' },
      String(source.start).slice(0, 10),
      String(source.end).slice(0, 10)
    ])
  ));

  const splits = [
    ['RANDOM SPLIT', artifact.random_split],
    ['TEMPORAL SPLIT', artifact.temporal_split]
  ].filter(([, value]) => value);

  if (splits.length) {
    container.appendChild(el('p', {
      class: 'eyebrow',
      text: 'SAME ARCHITECTURE, SAME HYPERPARAMETERS, DIFFERENT SPLIT'
    }));
    container.appendChild(table(
      ['SPLIT', 'TRAIN ROWS', 'TEST ROWS', 'TEST SPAM RATE', 'TEST WINDOW', 'ACCURACY', 'PRECISION', 'RECALL', 'F1', 'FPR', 'FNR', 'PR-AUC', 'ROC-AUC'],
      splits.map(([label, payload]) => {
        const metrics = payload.metrics || {};
        const test = payload.test || {};
        return [
          label,
          integer((payload.train || {}).rows),
          integer(test.rows),
          test.spam_ratio === undefined || test.spam_ratio === null ? 'Not available' : pct(test.spam_ratio, 1),
          test.start ? `${String(test.start).slice(0, 10)} to ${String(test.end).slice(0, 10)}` : 'Not available',
          { text: pct(metrics.accuracy, 2), class: 'num' },
          { text: pct(metrics.precision, 2), class: 'num' },
          { text: pct(metrics.recall, 2), class: 'num' },
          { text: pct(metrics.f1, 2), class: 'num' },
          { text: pct(metrics.false_positive_rate, 2), class: 'num' },
          { text: pct(metrics.false_negative_rate, 2), class: 'num' },
          { text: pct(metrics.pr_auc, 2), class: 'num' },
          { text: pct(metrics.roc_auc, 2), class: 'num' }
        ];
      })
    ));
  }

  const comparison = artifact.comparison;
  if (comparison) {
    container.appendChild(kvGrid([
      ['F1: TEMPORAL MINUS RANDOM', signed(comparison.f1_difference, 4),
        comparison.f1_difference < 0 ? 'danger' : 'safe'],
      ['RECALL DIFFERENCE', signed(comparison.recall_difference, 4),
        comparison.recall_difference < 0 ? 'danger' : 'safe'],
      ['FALSE-POSITIVE-RATE DIFFERENCE', signed(comparison.false_positive_rate_difference, 4),
        comparison.false_positive_rate_difference > 0 ? 'danger' : 'safe'],
      ['PR-AUC DIFFERENCE', signed(comparison.pr_auc_difference, 4),
        comparison.pr_auc_difference < 0 ? 'danger' : 'safe']
    ]));
  }

  const eras = artifact.era_evaluation || [];
  if (eras.length) {
    container.appendChild(el('p', {
      class: 'eyebrow',
      text: 'DELIVERED MODEL EVALUATED PER ERA (NOT RETRAINED)'
    }));
    container.appendChild(table(
      ['SOURCE', 'ERA', 'ROWS', 'SPAM', 'HAM', 'RECALL', 'PRECISION', 'F1', 'FALSE-POSITIVE RATE', 'MEASURABLE METRICS'],
      eras.map((era) => {
        const metrics = era.metrics || {};
        return [
          era.source,
          era.era,
          String(era.rows),
          String(era.spam),
          String(era.ham),
          { text: metrics.recall === undefined ? 'Not available' : pct(metrics.recall, 2), class: 'num' },
          { text: metrics.precision === undefined ? 'Not available' : pct(metrics.precision, 2), class: 'num' },
          { text: metrics.f1 === undefined ? 'Not available' : pct(metrics.f1, 2), class: 'num' },
          { text: metrics.false_positive_rate === undefined ? 'Not available' : pct(metrics.false_positive_rate, 2), class: 'num' },
          era.metric_note || 'all metrics'
        ];
      })
    ));
  }

  (artifact.limitations || []).forEach((limitation) => container.appendChild(bullet(limitation, 'alert')));
}

function signed(value, digits = 4) {
  if (value === null || value === undefined) return 'Not available';
  return `${value > 0 ? '+' : ''}${Number(value).toFixed(digits)}`;
}

export function renderPerformance(artifact) {
  const available = artifact && artifact.available === true;
  show('performance-empty', !available);
  const body = document.getElementById('performance-body');
  body.hidden = !available;
  const chip = document.getElementById('performance-chip');
  chip.textContent = available ? 'GENERATED' : 'NOT GENERATED';
  chip.className = `chip ${available ? 'chip-artifact' : 'chip-danger'}`;
  if (!available) return;

  const container = document.getElementById('performance-body');
  mount(container);

  const model = artifact.model || {};
  const environment = artifact.environment || {};
  container.appendChild(kvGrid([
    ['ARTIFACT SIZE', model.artifact_bytes ? `${(model.artifact_bytes / 1024 / 1024).toFixed(2)} MB` : 'Not available'],
    ['SEQUENCE LENGTH', integer(model.sequence_length)],
    ['VOCABULARY SIZE', integer(model.vocabulary_size)],
    ['PYTHON', environment.python || 'Not available'],
    ['TENSORFLOW', environment.tensorflow || 'Not available'],
    ['PLATFORM', environment.platform || 'Not available'],
    ['GENERATED', artifact.generated_at || 'Not available']
  ]));

  const latency = artifact.latency || {};
  container.appendChild(table(
    ['MEASUREMENT', 'SAMPLES', 'MEAN', 'MEDIAN', 'P95', 'MIN', 'MAX'],
    Object.entries(latency).map(([name, stats]) => [
      name.replace(/_/g, ' '),
      integer(stats.samples),
      { text: ms(stats.mean_ms), class: 'num' },
      { text: ms(stats.median_ms), class: 'num' },
      { text: ms(stats.p95_ms), class: 'num' },
      { text: ms(stats.min_ms), class: 'num' },
      { text: ms(stats.max_ms), class: 'num' }
    ])
  ));

  const optimisations = artifact.optimisations || [];
  if (optimisations.length) {
    container.appendChild(el('p', { class: 'eyebrow', text: 'BEFORE / AFTER OPTIMISATION' }));
    container.appendChild(table(
      ['OPTIMISATION', 'BEFORE', 'AFTER', 'SPEEDUP', 'NOTE'],
      optimisations.map((item) => [
        item.name,
        ms(item.before_ms),
        ms(item.after_ms),
        { text: item.speedup ? `${item.speedup.toFixed(2)}x` : 'n/a', class: 'num' },
        item.note || ''
      ])
    ));
  }

  const equivalence = artifact.equivalence;
  if (equivalence) {
    container.appendChild(kvGrid([
      ['MAX |PROBABILITY DIFFERENCE| vs Model.predict', Number(equivalence.max_absolute_difference).toExponential(3)],
      ['DECISIONS CHANGED BY OPTIMISATION', integer(equivalence.decisions_changed),
        equivalence.decisions_changed ? 'danger' : 'safe'],
      ['PAIRS COMPARED', integer(equivalence.samples)]
    ]));
  }

  const sequenceStudy = artifact.sequence_length_study || [];
  if (sequenceStudy.length) {
    container.appendChild(el('p', { class: 'eyebrow', text: 'SEQUENCE-LENGTH STUDY (MEASURED, NOT ADOPTED AUTOMATICALLY)' }));
    container.appendChild(table(
      ['SEQUENCE LENGTH', 'ACCURACY', 'PRECISION', 'RECALL', 'F1', 'FPR', 'PR-AUC', 'MEAN INFERENCE'],
      sequenceStudy.map((row) => [
        String(row.sequence_length),
        { text: pct(row.accuracy, 3), class: 'num' },
        { text: pct(row.precision, 3), class: 'num' },
        { text: pct(row.recall, 3), class: 'num' },
        { text: pct(row.f1, 3), class: 'num' },
        { text: pct(row.false_positive_rate, 3), class: 'num' },
        { text: pct(row.pr_auc, 3), class: 'num' },
        { text: ms(row.mean_inference_ms), class: 'num' }
      ])
    ));
  }

  const memory = artifact.memory || {};
  container.appendChild(kvGrid([
    ['PROCESS RSS', memory.peak_rss_mb ? `${Number(memory.peak_rss_mb).toFixed(1)} MB` : 'Not available'],
    ['MEMORY NOTE', memory.note || 'Not available', 'muted']
  ]));
}

export function renderHealth(health) {
  const container = document.getElementById('health-body');
  if (!health || health.available === false) {
    mount(container, bullet((health && health.reason) || 'Model artifacts are not present.', 'alert'));
    return;
  }
  const model = health.model || {};
  const live = health.live_measurement || {};
  const latest = health.latest_analysis;
  const latestRows = latest
    ? [
      ['LAST EMAIL RESULT', latest.prediction, latest.prediction === 'SPAM' ? 'danger' : 'safe'],
      ['LAST EMAIL INFERENCE', ms(latest.inference_ms)],
      ['LAST EMAIL TOTAL REQUEST', ms(latest.total_ms)],
      ['LAST EMAIL MODEL TOKENS', integer(latest.token_count)],
      ['LAST EMAIL UNFAMILIAR TOKENS', pct(latest.oov_rate, 1)],
      ['LAST EMAIL SAFETY STABILITY', latest.prediction_stability === null ? 'not requested' : pct(latest.prediction_stability, 1)]
    ]
    : [['LAST EMAIL', 'Analyze an email to populate email-specific system values.', 'muted']];
  mount(container,
    kvGrid([
      ['ARTIFACT SIZE', model.artifact_bytes ? `${(model.artifact_bytes / 1024 / 1024).toFixed(2)} MB` : 'Not available'],
      ['SEQUENCE LENGTH', integer(model.sequence_length)],
      ['VOCABULARY SIZE', integer(model.vocabulary_size)],
      ['ARCHITECTURE', model.architecture || 'Not available'],
      ['TRAINING CORPUS', model.dataset || 'Not available'],
      ['FIRST-CALL TRACING (PAID AT LOAD)', ms(model.warmup_latency_ms)],
      ['SINGLE REQUEST, END TO END', ms(live.sequential_per_item_ms)],
      ['BATCHED PER ITEM', ms(live.batched_per_item_ms)],
      ['BATCH SPEEDUP', live.speedup_factor ? `${live.speedup_factor.toFixed(2)}x` : 'Not available', 'safe']
    ]),
    el('p', { class: 'eyebrow', text: 'MOST RECENT EMAIL (METADATA ONLY)' }),
    kvGrid(latestRows),
    null
  );
}

export function renderArtifacts(inventory) {
  const container = document.getElementById('artifact-table');
  const entries = Object.entries(inventory || {});
  const present = entries.filter(([, value]) => value.present).length;
  const chip = document.getElementById('artifact-chip');
  if (chip) chip.textContent = `${present} / ${entries.length} PRESENT`;
  mount(container, table(
    ['ARTIFACT TYPE', 'FILE', 'PRESENT', 'SIZE'],
    entries.map(([key, value]) => [
      key,
      { text: value.file, mono: true },
      value.present ? tag('present', 'safe') : tag('absent', 'danger'),
      { text: value.present && value.bytes ? `${(value.bytes / 1024).toFixed(1)} KB` : 'Not available', class: 'num' }
    ])
  ));
}

export function renderHistory(history) {
  const container = document.getElementById('history-table');
  mount(container,
    kvGrid([
      ['ANALYZED', integer(history.analyzed)],
      ['SPAM', integer(history.spam_count)],
      ['NOT SPAM', integer(history.legitimate_count)]
    ]),
    table(
      ['CHECKED AT', 'RESULT'],
      (history.entries || []).map((entry) => [
        new Date(entry.timestamp).toLocaleString(),
        entry.prediction === 'SPAM' ? tag('SPAM', 'danger') : tag('NOT SPAM', 'safe')
      ]),
      { empty: 'No emails have been analysed in this session.' }
    )
  );
}
