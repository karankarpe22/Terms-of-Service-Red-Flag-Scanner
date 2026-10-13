/**
 * ToS Red-Flag Scanner - Frontend Application Logic
 * Technology: HTML5, CSS3, Vanilla JavaScript (ES2022)
 *
 * Architecture:
 * - Deterministic, reactive state management
 * - Strict evidence-first presentation
 * - Full communication with FastAPI backend endpoints
 * - Graceful degradation when Gemini is unavailable
 */

// Configurable API Base URL
const API_BASE_URL = (window.location.protocol === 'file:')
  ? 'http://127.0.0.1:8000/api'
  : `${window.location.origin}/api`;

// Application State
const state = {
  inputMode: 'pdf', // 'pdf' | 'text'
  selectedFile: null,
  pastedText: '',
  docTitle: '',
  selectedConcerns: [],
  documentId: null,
  documentData: null,
  clauses: [],
  relationships: [],
  synthesis: null,
  isAnalyzing: false,
  
  // Filtering & View State
  activeCategoryFilter: 'ALL',
  activeAttentionFilter: 'ALL',
  searchQuery: '',
  
  // Cache for explanations to avoid duplicate calls
  clauseExplanationCache: {},
  relationshipExplanationCache: {},
  qaCache: {},
  
  // Currently opened in modal
  selectedClause: null,
  selectedRelationship: null,
};


// Standard Sample Terms for instant 1-click viva testing
const SAMPLE_TOS_TEXT = `Sample Online Service - Terms of Service

1. Acceptance of Terms
By accessing or using our service, you agree to be bound by these Terms of Service. If you do not agree, do not use the service.

2. User Accounts and Termination
You may terminate your account at any time through your account settings. However, we reserve the right to suspend or terminate your account immediately, without notice or liability, for any reason whatsoever.

3. Payment, Billing, and Auto-Renewal
All paid plans are billed in advance on a recurring monthly or annual basis. Your subscription will automatically renew at the end of each billing cycle unless you cancel it at least thirty (30) days prior to the expiration date. All subscription fees paid are non-refundable.

4. Unilateral Modifications
We reserve the right, at our sole discretion, to modify or replace these Terms and modify pricing at any time without prior individual notice. Your continued use of the service following the posting of any changes constitutes binding acceptance of those changes.

5. User Content and License Grant
By submitting or posting content on or through the service, you grant us a worldwide, perpetual, irrevocable, royalty-free, transferable, and sublicensable license to use, reproduce, modify, adapt, publish, translate, and distribute such content in any media.

6. Limitation of Liability
To the maximum extent permitted by applicable law, in no event shall the company or its suppliers be liable for any special, incidental, indirect, or consequential damages whatsoever, and total aggregate liability shall not exceed fifty US dollars ($50.00).

7. Dispute Resolution and Class Action Waiver
Any dispute arising out of or relating to these Terms shall be resolved exclusively through final and binding individual arbitration. You expressly waive any right to participate in a class action lawsuit or class-wide arbitration against the company.

8. Data Retention Following Account Closure
Upon termination of your account, we may retain your personal data, usage logs, and content for a period of up to seven (7) years to comply with regulatory obligations and enforce our agreements.`;

// DOM Element References
let dom = {};

function bootstrap() {
  initDomReferences();
  initEventListeners();
  updateAnalyzeButtonState();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrap);
} else {
  bootstrap();
}

function initDomReferences() {
  dom = {
    // Header & Actions
    btnNewScan: document.getElementById('btnNewScan'),
    globalAlert: document.getElementById('globalAlert'),
    alertTitle: document.getElementById('alertTitle'),
    alertMessage: document.getElementById('alertMessage'),
    btnDismissAlert: document.getElementById('btnDismissAlert'),

    // Sections
    inputSection: document.getElementById('inputSection'),
    progressSection: document.getElementById('progressSection'),
    resultsSection: document.getElementById('resultsSection'),

    // Input Tabs
    tabPdf: document.getElementById('tabPdf'),
    tabText: document.getElementById('tabText'),
    pdfPanel: document.getElementById('pdfPanel'),
    textPanel: document.getElementById('textPanel'),

    // PDF Controls
    dropZone: document.getElementById('dropZone'),
    pdfFileInput: document.getElementById('pdfFileInput'),
    selectedFileCard: document.getElementById('selectedFileCard'),
    selectedFileName: document.getElementById('selectedFileName'),
    selectedFileSize: document.getElementById('selectedFileSize'),
    btnRemoveFile: document.getElementById('btnRemoveFile'),

    // Text Controls
    docTitleInput: document.getElementById('docTitleInput'),
    tosTextInput: document.getElementById('tosTextInput'),
    charCount: document.getElementById('charCount'),
    btnLoadSample: document.getElementById('btnLoadSample'),
    btnClearText: document.getElementById('btnClearText'),

    // Concerns & Analyze
    btnAnalyze: document.getElementById('btnAnalyze'),

    // Progress Steps
    step1: document.getElementById('step1'),
    step2: document.getElementById('step2'),
    step3: document.getElementById('step3'),
    step4: document.getElementById('step4'),

    // Results Summary
    resSourceType: document.getElementById('resSourceType'),
    resDocId: document.getElementById('resDocId'),
    resClauseBadge: document.getElementById('resClauseBadge'),
    resDocTitle: document.getElementById('resDocTitle'),
    btnReanalyze: document.getElementById('btnReanalyze'),
    synthesisSummary: document.getElementById('synthesisSummary'),
    synthesisKeyPoints: document.getElementById('synthesisKeyPoints'),
    synthesisDegradedNotice: document.getElementById('synthesisDegradedNotice'),

    // Visual Gauge & Risk Metrics
    gaugeValueArc: document.getElementById('gaugeValueArc'),
    gaugeScoreNumber: document.getElementById('gaugeScoreNumber'),
    riskVerdictBadge: document.getElementById('riskVerdictBadge'),
    riskTrapRatio: document.getElementById('riskTrapRatio'),
    riskCompoundScore: document.getElementById('riskCompoundScore'),

    // Stat Cards & Visual Tracks
    countHigh: document.getElementById('countHigh'),
    countMedium: document.getElementById('countMedium'),
    countLow: document.getElementById('countLow'),
    countInfo: document.getElementById('countInfo'),
    countRels: document.getElementById('countRels'),
    cardMetricHigh: document.getElementById('cardMetricHigh'),
    cardMetricMedium: document.getElementById('cardMetricMedium'),
    cardMetricLow: document.getElementById('cardMetricLow'),
    cardMetricInfo: document.getElementById('cardMetricInfo'),
    cardMetricRels: document.getElementById('cardMetricRels'),
    barMetricHigh: document.getElementById('barMetricHigh'),
    barMetricMedium: document.getElementById('barMetricMedium'),
    barMetricLow: document.getElementById('barMetricLow'),
    barMetricInfo: document.getElementById('barMetricInfo'),
    barMetricRels: document.getElementById('barMetricRels'),
    pctMetricHigh: document.getElementById('pctMetricHigh'),
    pctMetricMedium: document.getElementById('pctMetricMedium'),
    pctMetricLow: document.getElementById('pctMetricLow'),
    pctMetricInfo: document.getElementById('pctMetricInfo'),
    pctMetricRels: document.getElementById('pctMetricRels'),

    // Contract Anatomy Heatmap
    contractHeatmapBar: document.getElementById('contractHeatmapBar'),
    heatmapTooltip: document.getElementById('heatmapTooltip'),

    // Visual Category Donut & Risk Matrix
    categoryDonutSvg: document.getElementById('categoryDonutSvg'),
    donutSlicesGroup: document.getElementById('donutSlicesGroup'),
    donutCenterCount: document.getElementById('donutCenterCount'),
    donutCenterLabel: document.getElementById('donutCenterLabel'),
    categoryLegendList: document.getElementById('categoryLegendList'),
    riskMatrixList: document.getElementById('riskMatrixList'),

    // Cross-Clause Network Visualization
    btnRelViewList: document.getElementById('btnRelViewList'),
    btnRelViewGraph: document.getElementById('btnRelViewGraph'),
    relGraphContainer: document.getElementById('relGraphContainer'),
    relNetworkCanvas: document.getElementById('relNetworkCanvas'),
    relGraphTooltip: document.getElementById('relGraphTooltip'),
    btnResetGraph: document.getElementById('btnResetGraph'),

    // Filters
    clauseSearchInput: document.getElementById('clauseSearchInput'),
    categoryFilter: document.getElementById('categoryFilter'),
    attentionFilter: document.getElementById('attentionFilter'),
    btnResetFilters: document.getElementById('btnResetFilters'),
    clauseCountLabel: document.getElementById('clauseCountLabel'),
    clausesContainer: document.getElementById('clausesContainer'),
    relationshipsContainer: document.getElementById('relationshipsContainer'),

    // Q&A (Phase 8)
    qaSection: document.getElementById('qaSection'),
    qaQuestionInput: document.getElementById('qaQuestionInput'),
    btnClearQuestion: document.getElementById('btnClearQuestion'),
    btnAskQuestion: document.getElementById('btnAskQuestion'),
    qaLoader: document.getElementById('qaLoader'),
    qaResultCard: document.getElementById('qaResultCard'),
    qaResultTag: document.getElementById('qaResultTag'),
    qaConfidenceMeta: document.getElementById('qaConfidenceMeta'),
    qaAnswerText: document.getElementById('qaAnswerText'),
    qaSourcesWrap: document.getElementById('qaSourcesWrap'),
    qaSourcesList: document.getElementById('qaSourcesList'),
    qaInsufficientNotice: document.getElementById('qaInsufficientNotice'),
    qaInsufficientMsg: document.getElementById('qaInsufficientMsg'),
    qaDegradedNotice: document.getElementById('qaDegradedNotice'),
    qaDegradedMsg: document.getElementById('qaDegradedMsg'),

    // Clause Modal
    clauseModal: document.getElementById('clauseModal'),

    modalClauseTitle: document.getElementById('modalClauseTitle'),
    modalCategoryBadge: document.getElementById('modalCategoryBadge'),
    modalAttentionBadge: document.getElementById('modalAttentionBadge'),
    modalSourceLocation: document.getElementById('modalSourceLocation'),
    modalOriginalText: document.getElementById('modalOriginalText'),
    modalAttentionReasons: document.getElementById('modalAttentionReasons'),
    modalAILoader: document.getElementById('modalAILoader'),
    modalAISummary: document.getElementById('modalAISummary'),
    modalAIAttentionReason: document.getElementById('modalAIAttentionReason'),
    modalAIKeyPointsWrap: document.getElementById('modalAIKeyPointsWrap'),
    modalAIKeyPoints: document.getElementById('modalAIKeyPoints'),
    modalAIEvidenceWrap: document.getElementById('modalAIEvidenceWrap'),
    modalAICitations: document.getElementById('modalAICitations'),
    modalAIUncertaintyWrap: document.getElementById('modalAIUncertaintyWrap'),
    modalAIUncertainty: document.getElementById('modalAIUncertainty'),
    modalRelationshipsBlock: document.getElementById('modalRelationshipsBlock'),
    modalRelatedList: document.getElementById('modalRelatedList'),
    btnRefreshClauseAI: document.getElementById('btnRefreshClauseAI'),
    btnCloseModal: document.getElementById('btnCloseModal'),
    btnModalCloseAction: document.getElementById('btnModalCloseAction'),

    // Relationship Modal
    relModal: document.getElementById('relModal'),
    modalRelType: document.getElementById('modalRelType'),
    modalRelHeadline: document.getElementById('modalRelHeadline'),
    modalRelConfidence: document.getElementById('modalRelConfidence'),
    relDiagSourceTitle: document.getElementById('relDiagSourceTitle'),
    relDiagSourceExcerpt: document.getElementById('relDiagSourceExcerpt'),
    relDiagArrowLabel: document.getElementById('relDiagArrowLabel'),
    relDiagTargetTitle: document.getElementById('relDiagTargetTitle'),
    relDiagTargetExcerpt: document.getElementById('relDiagTargetExcerpt'),
    modalRelRationale: document.getElementById('modalRelRationale'),
    modalRelTrigger: document.getElementById('modalRelTrigger'),
    btnRequestRelAI: document.getElementById('btnRequestRelAI'),
    relAILoader: document.getElementById('relAILoader'),
    relAIContent: document.getElementById('relAIContent'),
    modalRelAISummary: document.getElementById('modalRelAISummary'),
    modalRelAIEffect: document.getElementById('modalRelAIEffect'),
    modalRelAICitations: document.getElementById('modalRelAICitations'),
    btnCloseRelModal: document.getElementById('btnCloseRelModal'),
    btnRelModalCloseAction: document.getElementById('btnRelModalCloseAction'),
  };
}

function initEventListeners() {
  // Tabs
  dom.tabPdf.addEventListener('click', () => switchInputTab('pdf'));
  dom.tabText.addEventListener('click', () => switchInputTab('text'));

  // Drag & Drop PDF
  ['dragenter', 'dragover'].forEach(name => {
    dom.dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dom.dropZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dom.dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dom.dropZone.classList.remove('drag-over');
    });
  });

  dom.dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleSelectedPdf(files[0]);
    }
  });

  // Click & Keyboard triggers on Drop Zone
  dom.dropZone.addEventListener('click', (e) => {
    if (e.target !== dom.pdfFileInput) {
      dom.pdfFileInput.click();
    }
  });

  dom.dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      dom.pdfFileInput.click();
    }
  });

  dom.pdfFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedPdf(e.target.files[0]);
    }
  });

  dom.btnRemoveFile.addEventListener('click', removeSelectedFile);

  // Text Inputs (Support input, paste, change, keyup)
  const syncTextInput = () => {
    state.pastedText = dom.tosTextInput.value;
    dom.charCount.textContent = `${state.pastedText.length.toLocaleString()} characters`;
    updateAnalyzeButtonState();
  };

  ['input', 'paste', 'change', 'keyup'].forEach(evt => {
    dom.tosTextInput.addEventListener(evt, () => {
      setTimeout(syncTextInput, 0);
    });
  });

  dom.docTitleInput.addEventListener('input', () => {
    state.docTitle = dom.docTitleInput.value.trim();
  });

  dom.btnLoadSample.addEventListener('click', () => {
    dom.tosTextInput.value = SAMPLE_TOS_TEXT;
    dom.docTitleInput.value = 'Sample Online Service Terms of Service';
    state.pastedText = SAMPLE_TOS_TEXT;
    state.docTitle = dom.docTitleInput.value;
    dom.charCount.textContent = `${state.pastedText.length.toLocaleString()} characters`;
    updateAnalyzeButtonState();
  });

  dom.btnClearText.addEventListener('click', () => {
    dom.tosTextInput.value = '';
    state.pastedText = '';
    dom.charCount.textContent = '0 characters';
    updateAnalyzeButtonState();
  });

  // Concerns Checkboxes
  document.querySelectorAll('.concern-chip input').forEach(input => {
    input.addEventListener('change', () => {
      const checked = Array.from(document.querySelectorAll('.concern-chip input:checked'))
        .map(i => i.value);
      state.selectedConcerns = checked;
    });
  });

  // Main Analyze Action
  dom.btnAnalyze.addEventListener('click', executeAnalysisPipeline);

  // Re-Analyze / New Scan
  dom.btnNewScan.addEventListener('click', resetToInputSection);
  dom.btnReanalyze.addEventListener('click', refreshExecutiveSynthesis);

  // Metric Card Clicks (Filter shortcuts with toggle support)
  dom.cardMetricHigh.addEventListener('click', () => {
    const isCur = state.activeAttentionFilter === 'High Attention' || state.activeAttentionFilter === 'HIGH';
    setAttentionFilter(isCur ? 'ALL' : 'High Attention');
  });
  dom.cardMetricMedium.addEventListener('click', () => {
    const isCur = state.activeAttentionFilter === 'Medium Attention' || state.activeAttentionFilter === 'MEDIUM';
    setAttentionFilter(isCur ? 'ALL' : 'Medium Attention');
  });
  dom.cardMetricLow.addEventListener('click', () => {
    const isCur = state.activeAttentionFilter === 'Low Attention' || state.activeAttentionFilter === 'LOW';
    setAttentionFilter(isCur ? 'ALL' : 'Low Attention');
  });
  dom.cardMetricInfo.addEventListener('click', () => {
    const isCur = state.activeAttentionFilter === 'Informational' || state.activeAttentionFilter === 'INFO';
    setAttentionFilter(isCur ? 'ALL' : 'Informational');
  });
  if (dom.cardMetricRels) {
    dom.cardMetricRels.addEventListener('click', () => {
      if (dom.relationshipsContainer) {
        dom.relationshipsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });
  }

  // Quick Risk Filter Pills
  document.querySelectorAll('.qf-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const target = pill.getAttribute('data-filter') || 'ALL';
      setAttentionFilter(target);
    });
  });

  // Filtering Controls
  dom.clauseSearchInput.addEventListener('input', (e) => {
    state.searchQuery = e.target.value.toLowerCase().trim();
    renderFilteredClauses();
  });

  dom.categoryFilter.addEventListener('change', (e) => {
    state.activeCategoryFilter = e.target.value;
    renderFilteredClauses();
  });

  dom.attentionFilter.addEventListener('change', (e) => {
    state.activeAttentionFilter = e.target.value;
    renderFilteredClauses();
  });

  dom.btnResetFilters.addEventListener('click', () => {
    dom.clauseSearchInput.value = '';
    dom.categoryFilter.value = 'ALL';
    dom.attentionFilter.value = 'ALL';
    state.searchQuery = '';
    state.activeCategoryFilter = 'ALL';
    state.activeAttentionFilter = 'ALL';
    renderFilteredClauses();
  });

  // Clause Modal
  dom.btnCloseModal.addEventListener('click', closeClauseModal);
  dom.btnModalCloseAction.addEventListener('click', closeClauseModal);
  dom.clauseModal.addEventListener('click', (e) => {
    if (e.target === dom.clauseModal) closeClauseModal();
  });
  dom.btnRefreshClauseAI.addEventListener('click', () => {
    if (state.selectedClause) {
      fetchClauseExplanation(state.selectedClause, true);
    }
  });

  // Relationship Modal
  dom.btnCloseRelModal.addEventListener('click', closeRelModal);
  dom.btnRelModalCloseAction.addEventListener('click', closeRelModal);
  dom.relModal.addEventListener('click', (e) => {
    if (e.target === dom.relModal) closeRelModal();
  });
  dom.btnRequestRelAI.addEventListener('click', () => {
    if (state.selectedRelationship) {
      fetchRelationshipExplanation(state.selectedRelationship);
    }
  });

  // Evidence-Grounded Q&A (Phase 8)
  if (dom.qaQuestionInput) {
    dom.qaQuestionInput.addEventListener('input', () => {
      const q = dom.qaQuestionInput.value.trim();
      dom.btnAskQuestion.disabled = (q.length === 0);
      dom.btnClearQuestion.classList.toggle('hidden', q.length === 0);
    });

    dom.qaQuestionInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !dom.btnAskQuestion.disabled) {
        e.preventDefault();
        const q = dom.qaQuestionInput.value.trim();
        if (q) handleAskQuestion(q);
      }
    });
  }

  if (dom.btnClearQuestion) {
    dom.btnClearQuestion.addEventListener('click', () => {
      dom.qaQuestionInput.value = '';
      dom.btnAskQuestion.disabled = true;
      dom.btnClearQuestion.classList.add('hidden');
      dom.qaQuestionInput.focus();
    });
  }

  if (dom.btnAskQuestion) {
    dom.btnAskQuestion.addEventListener('click', () => {
      const q = dom.qaQuestionInput.value.trim();
      if (q) handleAskQuestion(q);
    });
  }

  // Suggested Questions Chips
  document.querySelectorAll('.qa-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const q = btn.getAttribute('data-question');
      if (q) {
        dom.qaQuestionInput.value = q;
        dom.btnAskQuestion.disabled = false;
        dom.btnClearQuestion.classList.remove('hidden');
        handleAskQuestion(q);
      }
    });
  });

  // Keyboard shortcut for closing modals

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeClauseModal();
      closeRelModal();
    }
  });

  // Dismiss global alert
  dom.btnDismissAlert.addEventListener('click', hideAlert);
}

// ---------------------------------------------------------------------------
// Tab & Input Management
// ---------------------------------------------------------------------------
function switchInputTab(mode) {
  state.inputMode = mode;
  if (mode === 'pdf') {
    dom.tabPdf.classList.add('active');
    dom.tabPdf.setAttribute('aria-selected', 'true');
    dom.tabText.classList.remove('active');
    dom.tabText.setAttribute('aria-selected', 'false');
    dom.pdfPanel.classList.add('active');
    dom.pdfPanel.classList.remove('hidden');
    dom.pdfPanel.removeAttribute('hidden');
    dom.textPanel.classList.remove('active');
    dom.textPanel.classList.add('hidden');
    dom.textPanel.setAttribute('hidden', '');
  } else {
    dom.tabText.classList.add('active');
    dom.tabText.setAttribute('aria-selected', 'true');
    dom.tabPdf.classList.remove('active');
    dom.tabPdf.setAttribute('aria-selected', 'false');
    dom.textPanel.classList.add('active');
    dom.textPanel.classList.remove('hidden');
    dom.textPanel.removeAttribute('hidden');
    dom.pdfPanel.classList.remove('active');
    dom.pdfPanel.classList.add('hidden');
    dom.pdfPanel.setAttribute('hidden', '');
    setTimeout(() => {
      if (dom.tosTextInput) dom.tosTextInput.focus();
    }, 50);
  }
  updateAnalyzeButtonState();
}

function handleSelectedPdf(file) {
  hideAlert();
  if (!file) return;

  // Validation: Only PDF files accepted
  const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
  if (!isPdf) {
    showAlert('Unsupported File Type', 'Please select a valid machine-readable PDF file (.pdf). Image files, Word docs, and scanned archives are not supported.', 'danger');
    return;
  }

  // Max size: 25MB
  if (file.size > 25 * 1024 * 1024) {
    showAlert('File Too Large', 'The selected PDF exceeds the 25 MB file size limit. Please upload a smaller document.', 'danger');
    return;
  }

  state.selectedFile = file;
  dom.selectedFileName.textContent = file.name;
  dom.selectedFileSize.textContent = formatBytes(file.size);
  dom.selectedFileCard.classList.remove('hidden');
  updateAnalyzeButtonState();
}

function removeSelectedFile() {
  state.selectedFile = null;
  dom.pdfFileInput.value = '';
  dom.selectedFileCard.classList.add('hidden');
  updateAnalyzeButtonState();
}

function updateAnalyzeButtonState() {
  let canAnalyze = false;
  if (state.inputMode === 'pdf') {
    canAnalyze = state.selectedFile !== null;
  } else {
    canAnalyze = state.pastedText.trim().length >= 40;
  }
  dom.btnAnalyze.disabled = !canAnalyze;
}

// ---------------------------------------------------------------------------
// End-to-End Analysis Orchestration
// ---------------------------------------------------------------------------
async function executeAnalysisPipeline() {
  hideAlert();
  state.isAnalyzing = true;
  state.clauseExplanationCache = {};
  state.relationshipExplanationCache = {};

  // Show Progress View
  dom.inputSection.classList.add('hidden');
  dom.resultsSection.classList.add('hidden');
  dom.progressSection.classList.remove('hidden');
  resetProgressSteps();

  try {
    // Step 1: Upload / Submit Document
    activateStep(dom.step1);
    let uploadData;

    if (state.inputMode === 'pdf') {
      const formData = new FormData();
      formData.append('file', state.selectedFile);

      const res = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let msg = `Server error (${res.status})`;
        try {
          const err = await res.json();
          if (err && err.detail) {
            msg = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
          }
        } catch {
          if (res.status === 502) {
            msg = 'Server temporarily unavailable (502 Bad Gateway). The document may exceed free-tier memory, or the server is restarting.';
          } else if (res.status === 413) {
            msg = 'The uploaded file is too large (HTTP 413 Payload Too Large).';
          } else if (res.status === 504) {
            msg = 'Processing timed out (HTTP 504 Gateway Timeout).';
          } else if (res.statusText) {
            msg = `${res.statusText} (${res.status})`;
          }
        }
        throw new Error(msg);
      }
      uploadData = await res.json();

    } else {
      const title = state.docTitle || 'Pasted Terms of Service';
      const payload = { title, text: state.pastedText };

      const res = await fetch(`${API_BASE_URL}/documents/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        let msg = `Server error (${res.status})`;
        try {
          const err = await res.json();
          if (err && err.detail) {
            msg = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
          }
        } catch {
          if (res.status === 502) {
            msg = 'Server temporarily unavailable (502 Bad Gateway).';
          } else if (res.statusText) {
            msg = `${res.statusText} (${res.status})`;
          }
        }
        throw new Error(msg);
      }
      uploadData = await res.json();
    }

    markStepComplete(dom.step1);
    state.documentId = uploadData.document_id;
    state.documentData = uploadData;
    state.clauses = uploadData.clauses || [];

    // Step 2: Clause Segmentation & Attention Confirmation
    activateStep(dom.step2);
    // Backend already classified and assigned attention levels during ingestion
    await sleep(250); // Visual stability
    markStepComplete(dom.step2);

    // Step 3: Cross-Clause Relationship Analysis
    activateStep(dom.step3);
    const relRes = await fetch(`${API_BASE_URL}/documents/${state.documentId}/relationships`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_relationships: 100, min_confidence: 0.50 }),
    });

    if (!relRes.ok) {
      console.warn('Relationship analysis returned non-200, continuing with empty relationships.');
      state.relationships = [];
    } else {
      const relData = await relRes.json();
      state.relationships = relData.relationships || [];
    }
    markStepComplete(dom.step3);

    // Step 4: Evidence-Grounded Synthesis
    activateStep(dom.step4);
    const synPayload = state.selectedConcerns.length > 0
      ? { user_concerns: state.selectedConcerns }
      : {};

    const synRes = await fetch(`${API_BASE_URL}/documents/${state.documentId}/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(synPayload),
    });

    if (synRes.ok) {
      const synData = await synRes.json();
      state.synthesis = synData;
    } else {
      console.warn('Synthesis endpoint returned non-200, setting null synthesis.');
      state.synthesis = null;
    }
    markStepComplete(dom.step4);

    await sleep(200);

    // Render Dashboard and show Results View
    renderResultsDashboard();
    dom.progressSection.classList.add('hidden');
    dom.resultsSection.classList.remove('hidden');
    dom.btnNewScan.classList.remove('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });

  } catch (error) {
    console.error('Pipeline execution error:', error);
    dom.progressSection.classList.add('hidden');
    dom.inputSection.classList.remove('hidden');
    showAlert('Analysis Error', error.message || 'An error occurred during document processing.', 'danger');
  } finally {
    state.isAnalyzing = false;
  }
}

function resetProgressSteps() {
  [dom.step1, dom.step2, dom.step3, dom.step4].forEach(step => {
    step.classList.remove('active', 'completed');
  });
}

function activateStep(stepEl) {
  stepEl.classList.add('active');
}

function markStepComplete(stepEl) {
  stepEl.classList.remove('active');
  stepEl.classList.add('completed');
}

function resetToInputSection() {
  dom.resultsSection.classList.add('hidden');
  dom.progressSection.classList.add('hidden');
  dom.inputSection.classList.remove('hidden');
  dom.btnNewScan.classList.add('hidden');
  resetQAView();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function refreshExecutiveSynthesis() {
  if (!state.documentId) {
    resetToInputSection();
    return;
  }

  const originalBtnHtml = dom.btnReanalyze.innerHTML;
  dom.btnReanalyze.disabled = true;
  dom.btnReanalyze.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin-animation">
      <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
      <path d="M3 3v5h5"/>
      <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>
      <path d="M21 21v-5h-5"/>
    </svg>
    Refreshing...
  `;

  dom.synthesisDegradedNotice.classList.add('hidden');
  dom.synthesisSummary.textContent = 'Generating plain-English executive briefing with Gemini AI...';
  dom.synthesisKeyPoints.innerHTML = '';

  try {
    const synPayload = state.selectedConcerns && state.selectedConcerns.length > 0
      ? { user_concerns: state.selectedConcerns }
      : {};

    const synRes = await fetch(`${API_BASE_URL}/documents/${state.documentId}/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(synPayload),
    });

    if (synRes.ok) {
      const synData = await synRes.json();
      state.synthesis = synData;
    } else {
      console.warn('Synthesis returned non-200');
      state.synthesis = null;
    }
  } catch (err) {
    console.error('Failed to refresh synthesis:', err);
    state.synthesis = null;
  } finally {
    renderSynthesisBlock();
    dom.btnReanalyze.innerHTML = originalBtnHtml;
    dom.btnReanalyze.disabled = false;
  }
}

function resetQAView() {
  if (!dom.qaQuestionInput) return;
  dom.qaQuestionInput.value = '';
  dom.btnAskQuestion.disabled = true;
  dom.btnClearQuestion.classList.add('hidden');
  dom.qaLoader.classList.add('hidden');
  dom.qaResultCard.classList.add('hidden');
  state.qaCache = {};
}

// ---------------------------------------------------------------------------
// Evidence-Grounded Document Q&A (Phase 8)
// ---------------------------------------------------------------------------
async function handleAskQuestion(questionText) {
  const question = questionText ? questionText.trim() : '';
  if (!question || !state.documentId) return;

  // Check cache first for current session
  if (state.qaCache[question]) {
    renderQAResult(state.qaCache[question]);
    return;
  }

  // Set UI to loading state
  dom.qaLoader.classList.remove('hidden');
  dom.qaResultCard.classList.add('hidden');
  dom.btnAskQuestion.disabled = true;

  try {
    const res = await fetch(`${API_BASE_URL}/documents/${state.documentId}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: question,
        top_k: 5,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: ${res.status}`);
    }

    const data = await res.json();
    state.qaCache[question] = data;
    renderQAResult(data);

  } catch (error) {
    console.error('Q&A error:', error);
    renderQAResult({
      status: 'error',
      question: question,
      answer: null,
      evidence_sufficient: false,
      sources: [],
      uncertainty: error.message || 'Unable to process question.',
      message: error.message || 'Unable to process question at this time.',
    });
  } finally {
    dom.qaLoader.classList.add('hidden');
    dom.btnAskQuestion.disabled = false;
  }
}

function renderQAResult(data) {
  if (!data) return;

  dom.qaResultCard.classList.remove('hidden');

  // Reset sub-blocks
  dom.qaAnswerText.innerHTML = '';
  dom.qaSourcesList.innerHTML = '';
  dom.qaSourcesWrap.classList.add('hidden');
  dom.qaInsufficientNotice.classList.add('hidden');
  dom.qaDegradedNotice.classList.add('hidden');

  // 1. Insufficient Evidence
  if (data.status === 'insufficient_evidence' || !data.evidence_sufficient) {
    dom.qaResultTag.className = 'qa-result-tag tag-insufficient';
    dom.qaResultTag.textContent = 'INSUFFICIENT EVIDENCE';
    dom.qaConfidenceMeta.textContent = 'Evidence Abstention (No Speculation)';
    
    dom.qaInsufficientNotice.classList.remove('hidden');
    dom.qaInsufficientMsg.textContent = data.message || 'The supplied Terms do not contain enough evidence to answer this question.';
    dom.qaAnswerText.textContent = 'Insufficient evidence in the supplied Terms to answer this question.';
    return;
  }

  // 2. LLM Unavailable
  if (data.status === 'llm_unavailable') {
    dom.qaResultTag.className = 'qa-result-tag tag-degraded';
    dom.qaResultTag.textContent = 'RETRIEVED EVIDENCE ONLY';
    dom.qaConfidenceMeta.textContent = 'Deterministic Retrieval';

    dom.qaDegradedNotice.classList.remove('hidden');
    dom.qaDegradedMsg.textContent = 'AI answer generation is currently unavailable. Retrieved source clauses are displayed below.';
    dom.qaAnswerText.textContent = 'AI answer generation is currently unavailable.';

    // Display retrieved sources
    if (data.sources && data.sources.length > 0) {
      renderQASources(data.sources);
    }
    return;
  }

  // 3. Validation Failed
  if (data.status === 'validation_failed') {
    dom.qaResultTag.className = 'qa-result-tag tag-insufficient';
    dom.qaResultTag.textContent = 'VALIDATION FAILED';
    dom.qaConfidenceMeta.textContent = 'Grounding Guardrail Intercept';

    dom.qaInsufficientNotice.classList.remove('hidden');
    dom.qaInsufficientMsg.textContent = data.uncertainty || 'The generated answer could not be verified against the supplied document evidence.';
    dom.qaAnswerText.textContent = 'The answer could not be verified against the contractual text.';
    return;
  }

  // 4. Success
  dom.qaResultTag.className = 'qa-result-tag';
  dom.qaResultTag.textContent = 'EVIDENCE-GROUNDED ANSWER';
  dom.qaConfidenceMeta.textContent = data.confidence ? `Algorithmic Confidence: ${(data.confidence * 100).toFixed(0)}%` : 'Retrieval Grounded';

  // Format Answer
  dom.qaAnswerText.innerHTML = formatQAAnswer(data.answer || 'No answer generated.');

  // Sources
  if (data.sources && data.sources.length > 0) {
    renderQASources(data.sources);
  }
}

function formatQAAnswer(rawAnswer) {
  return escapeHtml(rawAnswer)
    .replace(/^ANSWER$/gm, '<strong>ANSWER</strong>')
    .replace(/^WHAT THE TERMS SAY$/gm, '<strong style="display:block; margin-top: 10px;">WHAT THE TERMS SAY</strong>')
    .replace(/^SOURCE$/gm, '<strong style="display:block; margin-top: 10px;">SOURCE</strong>')
    .replace(/^UNCERTAINTY$/gm, '<strong style="display:block; margin-top: 10px;">UNCERTAINTY</strong>');
}

function renderQASources(sources) {
  dom.qaSourcesList.innerHTML = '';
  sources.forEach(src => {
    const card = document.createElement('div');
    card.className = 'qa-source-card';
    card.setAttribute('role', 'button');
    card.setAttribute('tabindex', '0');
    card.setAttribute('aria-label', `View details for clause ${src.clause_id}`);
    card.onclick = () => window.openClauseModal(src.clause_id);
    card.onkeydown = (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        window.openClauseModal(src.clause_id);
      }
    };

    card.innerHTML = `
      <div class="qa-source-header">
        <span class="qa-source-title">${escapeHtml(src.section_title || 'Contract Provision')}</span>
        <span class="qa-source-loc">${escapeHtml(src.source_location || '')}</span>
      </div>
      <p class="qa-source-quote">"${escapeHtml(src.quoted_text)}"</p>
      <div class="qa-source-action">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
          <polyline points="15 3 21 3 21 9"></polyline>
          <line x1="10" y1="14" x2="21" y2="3"></line>
        </svg>
        Click to inspect original clause in context
      </div>
    `;
    dom.qaSourcesList.appendChild(card);
  });
  dom.qaSourcesWrap.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// Results Dashboard Rendering
// ---------------------------------------------------------------------------

function renderResultsDashboard() {
  // Document Metadata
  const doc = state.documentData;
  dom.resDocTitle.textContent = doc.filename || 'Terms of Service';
  dom.resSourceType.textContent = doc.source_type === 'pdf' ? 'PDF Document' : 'Text Input';
  dom.resDocId.textContent = doc.document_id;

  // Counts Calculation
  const highCount = state.clauses.filter(c => c.attention_level === 'High Attention').length;
  const medCount = state.clauses.filter(c => c.attention_level === 'Medium Attention').length;
  const lowCount = state.clauses.filter(c => c.attention_level === 'Low Attention').length;
  const infoCount = state.clauses.filter(c => c.attention_level === 'Informational').length;
  const relsCount = state.relationships.length;

  dom.countHigh.textContent = highCount;
  dom.countMedium.textContent = medCount;
  dom.countLow.textContent = lowCount;
  dom.countInfo.textContent = infoCount;
  dom.countRels.textContent = relsCount;

  // Update Quick Filter labels
  const qfAll = document.getElementById('qfAll');
  const qfHigh = document.getElementById('qfHigh');
  const qfMedium = document.getElementById('qfMedium');
  const qfStandard = document.getElementById('qfStandard');
  if (qfAll) qfAll.textContent = `All Clauses (${state.clauses.length})`;
  if (qfHigh) qfHigh.textContent = `🚨 Critical Flags (${highCount})`;
  if (qfMedium) qfMedium.textContent = `⚠️ Notable Terms (${medCount})`;
  if (qfStandard) qfStandard.textContent = `ℹ️ Standard Terms (${lowCount + infoCount})`;

  // Update Total Clauses Badge
  if (dom.resClauseBadge) {
    dom.resClauseBadge.textContent = `${state.clauses.length} Clauses Analyzed`;
  }

  // Visual Analytics, Gauges & Donut Breakdown
  renderRiskGauge(highCount, medCount, lowCount, state.clauses.length, relsCount);
  renderMetricProgressBars(highCount, medCount, lowCount, infoCount, relsCount, state.clauses.length);
  renderContractHeatmap();
  renderCategoryDonut();
  renderRiskExposureMatrix();
  initRelationshipNetworkGraph();

  // Executive Synthesis
  renderSynthesisBlock();

  // Render Lists - Default to Critical Flags if any exist so user isn't overwhelmed!
  if (highCount > 0) {
    setAttentionFilter('High Attention');
  } else if (medCount > 0) {
    setAttentionFilter('Medium Attention');
  } else {
    setAttentionFilter('ALL');
  }
  renderRelationshipsList();
}

function renderSynthesisBlock() {
  dom.synthesisKeyPoints.innerHTML = '';
  const syn = state.synthesis;

  if (syn && syn.status === 'success' && syn.explanation) {
    dom.synthesisDegradedNotice.classList.add('hidden');
    dom.synthesisSummary.textContent = syn.explanation.summary || 'Summary generated.';
    
    if (syn.explanation.key_points && syn.explanation.key_points.length > 0) {
      syn.explanation.key_points.forEach(point => {
        const div = document.createElement('div');
        div.className = 'point-bullet';
        div.textContent = point;
        dom.synthesisKeyPoints.appendChild(div);
      });
    }
  } else if (syn && syn.status === 'llm_unavailable') {
    dom.synthesisDegradedNotice.classList.remove('hidden');
    dom.synthesisSummary.textContent = 'Deterministic analysis succeeded. AI-generated executive synthesis is offline (API key unconfigured or mock mode inactive).';
  } else {
    dom.synthesisDegradedNotice.classList.remove('hidden');
    dom.synthesisSummary.textContent = 'Identified contractual clauses and structural relationships based on deterministic risk indicators.';
  }
}

// ===========================================================================
// VISUAL DATA VISUALIZATION ENGINE
// ===========================================================================

// 1. Contract Risk Gauge (Radial SVG Gauge & Dynamic Score Meter)
function renderRiskGauge(highCount, medCount, lowCount, totalClauses, relsCount) {
  if (!dom.gaugeValueArc || !dom.gaugeScoreNumber) return;

  const total = Math.max(totalClauses, 1);
  const rawScore = Math.min(
    100,
    Math.round(
      ((highCount * 30 + medCount * 10 + lowCount * 2) / total) * 20 +
      Math.min(15, relsCount * 1.5)
    )
  );

  let score = rawScore;
  if (highCount === 0 && medCount === 0) {
    score = Math.min(15, rawScore);
  } else if (highCount > 0) {
    score = Math.max(65, rawScore);
  } else if (medCount > 0) {
    score = Math.max(38, Math.min(60, rawScore));
  }

  // Semi-circle circumference of radius 75 is ~235.6
  const fullArc = 235.6;
  const targetOffset = fullArc - (fullArc * (score / 100));
  dom.gaugeValueArc.style.strokeDasharray = `${fullArc}`;
  dom.gaugeValueArc.style.strokeDashoffset = `${targetOffset}`;

  // Animated Count-Up
  animateCountUp(dom.gaugeScoreNumber, score);

  // Risk Verdict Badge
  if (dom.riskVerdictBadge) {
    dom.riskVerdictBadge.className = 'risk-verdict-badge';
    if (score < 30) {
      dom.riskVerdictBadge.classList.add('verdict-low');
      dom.riskVerdictBadge.innerHTML = '<span>●</span> Standard Agreement';
    } else if (score <= 60) {
      dom.riskVerdictBadge.classList.add('verdict-moderate');
      dom.riskVerdictBadge.innerHTML = '<span>●</span> Moderate Caution';
    } else {
      dom.riskVerdictBadge.classList.add('verdict-high');
      dom.riskVerdictBadge.innerHTML = '<span>●</span> Critical Traps Detected';
    }
  }

  // Submetrics
  if (dom.riskTrapRatio) {
    const ratio = Math.round(((highCount + medCount) / total) * 100);
    dom.riskTrapRatio.textContent = `${ratio}%`;
  }
  if (dom.riskCompoundScore) {
    dom.riskCompoundScore.textContent = `${relsCount} traps`;
  }
}

function animateCountUp(element, target, duration = 800) {
  if (!element) return;
  const start = 0;
  const startTime = performance.now();
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + (target - start) * ease);
    element.textContent = current;
    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      element.textContent = target;
    }
  }
  requestAnimationFrame(update);
}

// 2. Metric Progress Bars
function renderMetricProgressBars(highCount, medCount, lowCount, infoCount, relsCount, totalClauses) {
  const total = Math.max(totalClauses, 1);
  const pHigh = Math.round((highCount / total) * 100);
  const pMed = Math.round((medCount / total) * 100);
  const pLow = Math.round((lowCount / total) * 100);
  const pInfo = Math.round((infoCount / total) * 100);

  if (dom.barMetricHigh) dom.barMetricHigh.style.width = `${pHigh}%`;
  if (dom.barMetricMedium) dom.barMetricMedium.style.width = `${pMed}%`;
  if (dom.barMetricLow) dom.barMetricLow.style.width = `${pLow}%`;
  if (dom.barMetricInfo) dom.barMetricInfo.style.width = `${pInfo}%`;
  if (dom.barMetricRels) dom.barMetricRels.style.width = `${Math.min(100, relsCount)}%`;

  if (dom.pctMetricHigh) dom.pctMetricHigh.textContent = `${pHigh}%`;
  if (dom.pctMetricMedium) dom.pctMetricMedium.textContent = `${pMed}%`;
  if (dom.pctMetricLow) dom.pctMetricLow.textContent = `${pLow}%`;
  if (dom.pctMetricInfo) dom.pctMetricInfo.textContent = `${pInfo}%`;
  if (dom.pctMetricRels) dom.pctMetricRels.textContent = `${relsCount}`;
}

// 3. Contract Anatomy / Document Risk Heatmap Strip
function renderContractHeatmap() {
  if (!dom.contractHeatmapBar) return;
  dom.contractHeatmapBar.innerHTML = '';
  const tooltip = dom.heatmapTooltip;

  state.clauses.forEach((c) => {
    const seg = document.createElement('div');
    seg.className = 'heatmap-segment';
    seg.setAttribute('data-clause-id', c.clause_id);

    if (c.attention_level === 'High Attention') {
      seg.style.backgroundColor = '#EF4444';
    } else if (c.attention_level === 'Medium Attention') {
      seg.style.backgroundColor = '#F59E0B';
    } else if (c.attention_level === 'Low Attention') {
      seg.style.backgroundColor = '#10B981';
    } else {
      seg.style.backgroundColor = '#3B82F6';
    }

    seg.addEventListener('mouseenter', (e) => {
      if (!tooltip) return;
      const title = c.section_title || `Clause ${c.clause_index}`;
      const level = c.attention_level || 'Standard';
      tooltip.innerHTML = `
        <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 2px;">#${c.clause_index}: ${escapeHtml(title)}</div>
        <div style="color: ${seg.style.backgroundColor}; font-weight: 600; font-size: 11px; margin-bottom: 4px;">${level}</div>
        <div style="color: var(--text-muted); font-size: 11px; max-height: 40px; overflow: hidden; text-overflow: ellipsis;">${escapeHtml(c.text.slice(0, 90))}...</div>
      `;
      tooltip.classList.remove('hidden');
      positionTooltip(e, tooltip);
    });

    seg.addEventListener('mousemove', (e) => {
      if (tooltip && !tooltip.classList.contains('hidden')) {
        positionTooltip(e, tooltip);
      }
    });

    seg.addEventListener('mouseleave', () => {
      if (tooltip) tooltip.classList.add('hidden');
    });

    seg.addEventListener('click', () => {
      scrollToClause(c.clause_id);
    });

    dom.contractHeatmapBar.appendChild(seg);
  });
}

function positionTooltip(e, tooltip) {
  const x = Math.min(window.innerWidth - 300, Math.max(10, e.clientX - 100));
  const y = e.clientY - 85;
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

function scrollToClause(clauseId) {
  const target = state.clauses.find(c => c.clause_id === clauseId);
  if (target) {
    if (state.activeAttentionFilter !== 'ALL' && state.activeAttentionFilter !== target.attention_level) {
      setAttentionFilter('ALL');
    }
    if (state.activeCategoryFilter !== 'ALL' && state.activeCategoryFilter !== target.primary_category) {
      state.activeCategoryFilter = 'ALL';
      if (dom.categoryFilter) dom.categoryFilter.value = 'ALL';
      renderFilteredClauses();
    }
  }

  setTimeout(() => {
    const el = document.getElementById(`clause-card-${clauseId}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      el.classList.add('highlight-pulse');
      setTimeout(() => el.classList.remove('highlight-pulse'), 2000);
    }
  }, 100);
}

// 4. Clause Category Breakdown (Interactive SVG Donut Chart)
const CATEGORY_COLORS = {
  'Privacy & Data': '#06B6D4',
  'Payment': '#10B981',
  'Dispute': '#EF4444',
  'Liability': '#F97316',
  'User Content': '#8B5CF6',
  'Changes': '#F59E0B',
  'Account': '#EC4899',
  'General': '#64748B',
};

function renderCategoryDonut() {
  if (!dom.donutSlicesGroup || !dom.categoryLegendList) return;
  dom.donutSlicesGroup.innerHTML = '';
  dom.categoryLegendList.innerHTML = '';

  const total = Math.max(state.clauses.length, 1);
  dom.donutCenterCount.textContent = state.clauses.length;
  dom.donutCenterLabel.textContent = 'Clauses';

  const counts = {};
  state.clauses.forEach(c => {
    const cat = c.primary_category || c.category || 'General';
    counts[cat] = (counts[cat] || 0) + 1;
  });

  const sortedCats = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const radius = 68;
  const circumference = 2 * Math.PI * radius;
  let accumulatedOffset = 0;

  sortedCats.forEach(([cat, count]) => {
    const fraction = count / total;
    const sliceLength = fraction * circumference;
    const color = CATEGORY_COLORS[cat] || '#818CF8';

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', '90');
    circle.setAttribute('cy', '90');
    circle.setAttribute('r', radius);
    circle.setAttribute('stroke', color);
    circle.setAttribute('stroke-dasharray', `${sliceLength} ${circumference - sliceLength}`);
    circle.setAttribute('stroke-dashoffset', `${-accumulatedOffset}`);
    circle.setAttribute('class', 'donut-slice');
    circle.setAttribute('data-category', cat);

    circle.addEventListener('mouseenter', () => {
      dom.donutCenterCount.textContent = count;
      dom.donutCenterLabel.textContent = cat;
      circle.style.stroke = '#FFFFFF';
    });
    circle.addEventListener('mouseleave', () => {
      dom.donutCenterCount.textContent = state.clauses.length;
      dom.donutCenterLabel.textContent = 'Clauses';
      circle.style.stroke = color;
    });

    circle.addEventListener('click', () => {
      filterByCategory(cat);
    });

    dom.donutSlicesGroup.appendChild(circle);
    accumulatedOffset += sliceLength;

    const legItem = document.createElement('div');
    legItem.className = 'category-legend-item';
    legItem.setAttribute('data-category', cat);
    legItem.innerHTML = `
      <div class="cat-legend-left">
        <span class="cat-color-dot" style="background: ${color}"></span>
        <span class="cat-name">${escapeHtml(cat)}</span>
      </div>
      <span class="cat-count-pill">${count} (${Math.round(fraction * 100)}%)</span>
    `;

    legItem.addEventListener('mouseenter', () => {
      circle.dispatchEvent(new Event('mouseenter'));
    });
    legItem.addEventListener('mouseleave', () => {
      circle.dispatchEvent(new Event('mouseleave'));
    });
    legItem.addEventListener('click', () => {
      filterByCategory(cat);
    });

    dom.categoryLegendList.appendChild(legItem);
  });
}

function filterByCategory(cat) {
  if (!dom.categoryFilter) return;
  dom.categoryFilter.value = cat;
  state.activeCategoryFilter = cat;
  renderFilteredClauses();
  if (dom.clausesContainer) {
    dom.clausesContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

// 5. Consumer Rights Exposure Matrix
function renderRiskExposureMatrix() {
  if (!dom.riskMatrixList) return;
  dom.riskMatrixList.innerHTML = '';

  const domains = [
    {
      id: 'unilateral',
      title: 'Unilateral Changes & Pricing',
      icon: '⚡',
      keywords: ['modify', 'replace', 'change', 'discretion', 'without notice'],
      categories: ['Changes'],
      desc: 'Can the company modify terms or raise prices without direct notice?',
    },
    {
      id: 'dispute',
      title: 'Forced Arbitration & Waivers',
      icon: '⚖️',
      keywords: ['arbitration', 'waive', 'class action', 'jury', 'dispute'],
      categories: ['Dispute'],
      desc: 'Mandatory individual arbitration and surrender of court litigation rights.',
    },
    {
      id: 'content',
      title: 'Content Rights & IP License',
      icon: '📜',
      keywords: ['license', 'royalty-free', 'perpetual', 'sublicense', 'derivative works'],
      categories: ['User Content'],
      desc: 'Scope of rights granted over user submissions, media, and data.',
    },
    {
      id: 'liability',
      title: 'Liability Shield & Disclaimers',
      icon: '🛡️',
      keywords: ['limitation of liability', 'damages', 'as is', 'consequential'],
      categories: ['Liability'],
      desc: 'Company immunity for service outages, data losses, or breaches.',
    },
  ];

  domains.forEach(d => {
    const matches = state.clauses.filter(c => {
      const catMatch = d.categories.includes(c.primary_category);
      const textMatch = d.keywords.some(k => (c.text || '').toLowerCase().includes(k));
      return catMatch || textMatch;
    });

    const highMatches = matches.filter(c => c.attention_level === 'High Attention').length;
    const medMatches = matches.filter(c => c.attention_level === 'Medium Attention').length;

    let status = 'Guarded';
    let pillClass = 'pill-guarded';
    let fillWidth = 20;
    let fillColor = '#10B981';

    if (highMatches > 0) {
      status = 'Flagged';
      pillClass = 'pill-flagged';
      fillWidth = 90;
      fillColor = '#EF4444';
    } else if (medMatches > 0 || matches.length > 2) {
      status = 'Notable';
      pillClass = 'pill-notable';
      fillWidth = 60;
      fillColor = '#F59E0B';
    }

    const item = document.createElement('div');
    item.className = 'risk-matrix-item';
    item.innerHTML = `
      <div class="matrix-item-header">
        <span class="matrix-item-title">${d.icon} ${d.title}</span>
        <span class="matrix-pill ${pillClass}">${status}</span>
      </div>
      <div class="matrix-bar-track">
        <div class="matrix-bar-fill" style="width: ${fillWidth}%; background: ${fillColor}"></div>
      </div>
      <div class="matrix-item-desc">${matches.length} provisions identified &bull; ${d.desc}</div>
    `;

    dom.riskMatrixList.appendChild(item);
  });
}

// 6. Interactive Cross-Clause Network Visualization
let graphAnimationId = null;
let graphNodes = [];
let graphLinks = [];
let hoveredNode = null;
let draggedNode = null;
let graphInitialized = false;

function initRelationshipNetworkGraph() {
  if (!dom.btnRelViewList || !dom.btnRelViewGraph) return;

  if (!graphInitialized) {
    graphInitialized = true;
    dom.btnRelViewList.addEventListener('click', () => {
      dom.btnRelViewList.classList.add('active');
      dom.btnRelViewGraph.classList.remove('active');
      dom.relationshipsContainer.classList.remove('hidden');
      dom.relGraphContainer.classList.add('hidden');
    });

    dom.btnRelViewGraph.addEventListener('click', () => {
      dom.btnRelViewGraph.classList.add('active');
      dom.btnRelViewList.classList.remove('active');
      dom.relationshipsContainer.classList.add('hidden');
      dom.relGraphContainer.classList.remove('hidden');
      buildNetworkGraphData();
    });

    if (dom.btnResetGraph) {
      dom.btnResetGraph.addEventListener('click', () => {
        buildNetworkGraphData();
      });
    }
  }

  // If graph is currently visible, rebuild data
  if (dom.relGraphContainer && !dom.relGraphContainer.classList.contains('hidden')) {
    buildNetworkGraphData();
  }
}

function buildNetworkGraphData() {
  const canvas = dom.relNetworkCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const nodeMap = new Map();
  state.relationships.forEach(r => {
    if (!nodeMap.has(r.source_clause_id)) {
      const c = state.clauses.find(item => item.clause_id === r.source_clause_id);
      if (c) nodeMap.set(r.source_clause_id, c);
    }
    if (!nodeMap.has(r.target_clause_id)) {
      const c = state.clauses.find(item => item.clause_id === r.target_clause_id);
      if (c) nodeMap.set(r.target_clause_id, c);
    }
  });

  const clausesList = Array.from(nodeMap.values()).slice(0, 24);
  const width = canvas.width;
  const height = canvas.height;
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.38;

  graphNodes = clausesList.map((c, i) => {
    const angle = (i / clausesList.length) * 2 * Math.PI;
    return {
      clause: c,
      id: c.clause_id,
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
      radius: c.attention_level === 'High Attention' ? 18 : (c.attention_level === 'Medium Attention' ? 16 : 14),
      color: c.attention_level === 'High Attention' ? '#EF4444' : (c.attention_level === 'Medium Attention' ? '#F59E0B' : '#06B6D4'),
    };
  });

  const nodeObjMap = new Map(graphNodes.map(n => [n.id, n]));

  graphLinks = state.relationships
    .filter(r => nodeObjMap.has(r.source_clause_id) && nodeObjMap.has(r.target_clause_id))
    .slice(0, 50)
    .map(r => ({
      source: nodeObjMap.get(r.source_clause_id),
      target: nodeObjMap.get(r.target_clause_id),
      rel: r,
    }));

  setupCanvasInteraction(canvas);
  startGraphAnimation(canvas, ctx);
}

function startGraphAnimation(canvas, ctx) {
  if (graphAnimationId) cancelAnimationFrame(graphAnimationId);

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connecting links
    graphLinks.forEach(link => {
      const isHighlighted = hoveredNode && (hoveredNode === link.source || hoveredNode === link.target);
      const isDimmed = hoveredNode && !isHighlighted;

      ctx.beginPath();
      ctx.moveTo(link.source.x, link.source.y);
      ctx.lineTo(link.target.x, link.target.y);

      let strokeColor = 'rgba(99, 102, 241, 0.25)';
      const type = (link.rel.relationship_type || '').toLowerCase();
      if (type.includes('contradict') || type.includes('conflict')) {
        strokeColor = isHighlighted ? 'rgba(239, 68, 68, 0.95)' : (isDimmed ? 'rgba(239, 68, 68, 0.08)' : 'rgba(239, 68, 68, 0.45)');
      } else if (type.includes('modify') || type.includes('override')) {
        strokeColor = isHighlighted ? 'rgba(245, 158, 11, 0.95)' : (isDimmed ? 'rgba(245, 158, 11, 0.08)' : 'rgba(245, 158, 11, 0.4)');
      } else {
        strokeColor = isHighlighted ? 'rgba(6, 182, 212, 0.95)' : (isDimmed ? 'rgba(6, 182, 212, 0.08)' : 'rgba(6, 182, 212, 0.35)');
      }

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = isHighlighted ? 2.5 : 1.2;
      ctx.stroke();
    });

    // Draw nodes
    graphNodes.forEach(node => {
      const isHovered = (hoveredNode === node);
      const isConnected = hoveredNode && graphLinks.some(l => (l.source === hoveredNode && l.target === node) || (l.target === hoveredNode && l.source === node));
      const isDimmed = hoveredNode && !isHovered && !isConnected;

      if (isHovered) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius + 6, 0, Math.PI * 2);
        ctx.fillStyle = node.color;
        ctx.globalAlpha = 0.35;
        ctx.fill();
        ctx.globalAlpha = 1.0;
      }

      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
      ctx.fillStyle = isDimmed ? '#1F2937' : node.color;
      ctx.fill();
      ctx.strokeStyle = isHovered ? '#FFFFFF' : '#111827';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = '#FFFFFF';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`C${node.clause.clause_index}`, node.x, node.y);
    });

    graphAnimationId = requestAnimationFrame(draw);
  }

  draw();
}

function setupCanvasInteraction(canvas) {
  const tooltip = dom.relGraphTooltip;

  function getNodeAt(x, y) {
    return graphNodes.find(n => {
      const dx = n.x - x;
      const dy = n.y - y;
      return (dx * dx + dy * dy) <= (n.radius * n.radius);
    });
  }

  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    if (draggedNode) {
      draggedNode.x = x;
      draggedNode.y = y;
      return;
    }

    const node = getNodeAt(x, y);
    hoveredNode = node || null;

    if (node && tooltip) {
      const c = node.clause;
      const conns = graphLinks.filter(l => l.source === node || l.target === node).length;
      tooltip.innerHTML = `
        <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 2px;">#${c.clause_index}: ${escapeHtml(c.section_title || 'General')}</div>
        <div style="color: ${node.color}; font-weight: 600; font-size: 11px; margin-bottom: 4px;">${c.attention_level} &bull; ${conns} Connections</div>
        <div style="color: var(--text-muted); font-size: 11px;">Click node to jump to clause text</div>
      `;
      tooltip.classList.remove('hidden');
      const tipX = Math.min(canvas.clientWidth - 220, Math.max(10, e.clientX - rect.left - 50));
      const tipY = Math.max(10, e.clientY - rect.top - 80);
      tooltip.style.left = `${tipX}px`;
      tooltip.style.top = `${tipY}px`;
    } else if (tooltip) {
      tooltip.classList.add('hidden');
    }
  };

  canvas.onmousedown = (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    const node = getNodeAt(x, y);
    if (node) {
      draggedNode = node;
    }
  };

  window.onmouseup = () => {
    draggedNode = null;
  };

  canvas.onclick = (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    const node = getNodeAt(x, y);
    if (node) {
      scrollToClause(node.clause.clause_id);
    }
  };

  canvas.onmouseleave = () => {
    hoveredNode = null;
    draggedNode = null;
    if (tooltip) tooltip.classList.add('hidden');
  };
}

// ---------------------------------------------------------------------------
// Clause Explorer & Card Rendering
// ---------------------------------------------------------------------------
function renderFilteredClauses() {
  const container = dom.clausesContainer;
  container.innerHTML = '';

  let filtered = state.clauses;

  // 1. Search Query Filter
  if (state.searchQuery) {
    const q = state.searchQuery;
    filtered = filtered.filter(c => 
      c.text.toLowerCase().includes(q) || 
      (c.section_title && c.section_title.toLowerCase().includes(q)) ||
      (c.attention_reasons && c.attention_reasons.some(r => r.toLowerCase().includes(q)))
    );
  }

  // 2. Category Filter
  if (state.activeCategoryFilter !== 'ALL') {
    filtered = filtered.filter(c => 
      c.primary_category === state.activeCategoryFilter ||
      c.category === state.activeCategoryFilter ||
      (c.secondary_categories && c.secondary_categories.includes(state.activeCategoryFilter))
    );
  }

  // 3. Attention Level Filter (Robust matching)
  if (state.activeAttentionFilter && state.activeAttentionFilter !== 'ALL') {
    const key = state.activeAttentionFilter.toUpperCase();
    if (key === 'STANDARD') {
      filtered = filtered.filter(c => {
        const lvl = (c.attention_level || '').toUpperCase();
        return lvl.includes('LOW') || lvl.includes('INFO');
      });
    } else {
      filtered = filtered.filter(c => {
        const lvl = (c.attention_level || '').toUpperCase();
        if (key.includes('HIGH')) return lvl.includes('HIGH');
        if (key.includes('MED')) return lvl.includes('MED');
        if (key.includes('LOW')) return lvl.includes('LOW');
        if (key.includes('INFO')) return lvl.includes('INFO');
        return lvl === key;
      });
    }
  }

  // Update counter label
  dom.clauseCountLabel.textContent = `Showing ${filtered.length} of ${state.clauses.length} clauses`;

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="empty-state" style="text-align: center; padding: 40px 20px; color: var(--text-muted);">
        <p>No clauses match the active filters.</p>
        <button class="btn btn-link" style="margin-top: 8px;" onclick="document.getElementById('btnResetFilters').click()">Clear Filters</button>
      </div>
    `;
    return;
  }

  // Render Cards
  filtered.forEach(clause => {
    const card = document.createElement('div');
    const attClass = getAttentionCardClass(clause.attention_level);
    card.className = `clause-card ${attClass}`;
    card.id = `clause-card-${clause.clause_id}`;
    card.setAttribute('role', 'listitem');

    // Relationships count for this clause
    const clauseRels = state.relationships.filter(
      r => r.source_clause_id === clause.clause_id || r.target_clause_id === clause.clause_id
    );

    // Attention badge
    const attBadge = getAttentionBadgeHtml(clause.attention_level);

    // Indicators
    let indicatorsHtml = '';
    let primaryTakeaway = '';
    if (clause.attention_reasons && clause.attention_reasons.length > 0) {
      primaryTakeaway = `<div class="card-takeaway-banner"><span class="takeaway-icon">🚨</span> <strong>Risk Flag:</strong> ${escapeHtml(clause.attention_reasons[0])}</div>`;
      const shown = clause.attention_reasons.slice(1, 3);
      if (shown.length > 0) {
        const chips = shown.map(r => `<span class="indicator-chip">${escapeHtml(r)}</span>`).join('');
        indicatorsHtml = `<div class="indicators-preview">${chips}</div>`;
      }
    } else {
      primaryTakeaway = `<div class="card-takeaway-banner standard"><span class="takeaway-icon">ℹ️</span> Standard contractual term (${escapeHtml(clause.primary_category || 'General')})</div>`;
    }

    const relTagHtml = clauseRels.length > 0 
      ? `<span class="rel-indicator-tag" title="Connected to other provisions">${clauseRels.length} ${clauseRels.length === 1 ? 'Relationship' : 'Relationships'}</span>`
      : '';

    card.innerHTML = `
      <div class="card-top-row">
        <div class="badge-row">
          <span class="tag-pill">${escapeHtml(clause.primary_category || clause.category || 'General')}</span>
          ${attBadge}
        </div>
        ${relTagHtml}
      </div>
      <h4 class="clause-title">${escapeHtml(clause.section_title || `Clause ${clause.clause_index}`)}</h4>
      ${primaryTakeaway}
      <p class="clause-excerpt"><span class="quote-label">Original text:</span> "${escapeHtml(clause.text.slice(0, 130))}${clause.text.length > 130 ? '...' : ''}"</p>
      ${indicatorsHtml}
      <div class="card-bottom-row">
        <span class="source-location-meta">Clause #${clause.clause_index} &bull; ${escapeHtml(clause.source_location || 'Document Body')}</span>
        <button class="btn btn-secondary btn-sm" onclick="openClauseModal('${clause.clause_id}')">
          Decode in Plain English &rarr;
        </button>
      </div>
    `;

    container.appendChild(card);
  });
}

function setAttentionFilter(level) {
  state.activeAttentionFilter = level;

  // Sync select dropdown
  if (dom.attentionFilter) {
    const upper = (level || '').toUpperCase();
    if (level === 'ALL') {
      dom.attentionFilter.value = 'ALL';
    } else {
      for (let opt of dom.attentionFilter.options) {
        const optVal = opt.value.toUpperCase();
        if (optVal === upper || (upper.includes('HIGH') && optVal.includes('HIGH')) || (upper.includes('MED') && optVal.includes('MED')) || (upper.includes('LOW') && optVal.includes('LOW')) || (upper.includes('INFO') && optVal.includes('INFO'))) {
          dom.attentionFilter.value = opt.value;
          break;
        }
      }
    }
  }

  // Update active highlight on metric cards
  [dom.cardMetricHigh, dom.cardMetricMedium, dom.cardMetricLow, dom.cardMetricInfo, dom.cardMetricRels].forEach(el => {
    if (el) el.classList.remove('active-metric');
  });

  const upper = (level || '').toUpperCase();
  if (upper.includes('HIGH') && dom.cardMetricHigh) dom.cardMetricHigh.classList.add('active-metric');
  else if (upper.includes('MED') && dom.cardMetricMedium) dom.cardMetricMedium.classList.add('active-metric');
  else if (upper.includes('LOW') && dom.cardMetricLow) dom.cardMetricLow.classList.add('active-metric');
  else if (upper.includes('INFO') && dom.cardMetricInfo) dom.cardMetricInfo.classList.add('active-metric');

  // Update active state on Quick Filter Pills
  document.querySelectorAll('.qf-pill').forEach(pill => {
    const pFilter = (pill.getAttribute('data-filter') || '').toUpperCase();
    if (level === 'ALL' && pFilter === 'ALL') {
      pill.classList.add('active');
    } else if (pFilter === upper || (upper.includes('HIGH') && pFilter.includes('HIGH')) || (upper.includes('MED') && pFilter.includes('MED')) || (upper === 'STANDARD' && pFilter === 'STANDARD')) {
      pill.classList.add('active');
    } else {
      pill.classList.remove('active');
    }
  });

  renderFilteredClauses();
}

// ---------------------------------------------------------------------------
// Clause Detail Modal & Evidence-First Breakdown
// ---------------------------------------------------------------------------
window.openClauseModal = async function(clauseId) {
  const clause = state.clauses.find(c => c.clause_id === clauseId);
  if (!clause) return;

  state.selectedClause = clause;

  // Populate deterministic metadata
  dom.modalClauseTitle.textContent = clause.section_title || `Clause ${clause.clause_index}`;
  dom.modalCategoryBadge.textContent = clause.primary_category || clause.category || 'General';
  dom.modalSourceLocation.textContent = `Clause #${clause.clause_index} • Location: ${clause.source_location || 'Document Body'}`;
  
  // Attention Badge
  dom.modalAttentionBadge.className = `attention-badge ${getAttentionBadgeClass(clause.attention_level)}`;
  dom.modalAttentionBadge.innerHTML = getAttentionBadgeInner(clause.attention_level);

  // Instant Plain-English Fallback (Zero-Waiting UX)
  const hasReasons = clause.attention_reasons && clause.attention_reasons.length > 0;
  dom.modalAISummary.textContent = hasReasons
    ? `Quick Take: ${clause.attention_reasons.join('. ')}.`
    : `Quick Take: This provision establishes standard terms for ${clause.primary_category || 'general service rules'}.`;
  dom.modalAIAttentionReason.textContent = hasReasons
    ? `Why it matters: This term gives the provider unilateral discretion or limits their responsibility towards you.`
    : `Standard contractual terms with no heightened one-sided risks detected.`;

  // 1. Original Source Evidence (Preserved completely)
  dom.modalOriginalText.textContent = clause.text;

  // 2. Deterministic Reasons
  dom.modalAttentionReasons.innerHTML = '';
  if (hasReasons) {
    clause.attention_reasons.forEach(r => {
      const li = document.createElement('li');
      li.textContent = r;
      dom.modalAttentionReasons.appendChild(li);
    });
  } else {
    const li = document.createElement('li');
    li.textContent = 'No heightened contractual indicators detected for this provision.';
    dom.modalAttentionReasons.appendChild(li);
  }

  // 3. Associated Relationships
  const relatedRels = state.relationships.filter(
    r => r.source_clause_id === clause.clause_id || r.target_clause_id === clause.clause_id
  );

  if (relatedRels.length > 0) {
    dom.modalRelationshipsBlock.classList.remove('hidden');
    dom.modalRelatedList.innerHTML = '';
    relatedRels.forEach(r => {
      const otherId = r.source_clause_id === clause.clause_id ? r.target_clause_id : r.source_clause_id;
      const otherClause = state.clauses.find(c => c.clause_id === otherId);
      const otherTitle = otherClause ? otherClause.section_title : otherId;

      const tag = document.createElement('div');
      tag.className = 'rel-card';
      tag.style.marginBottom = '6px';
      tag.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span class="rel-badge rel-badge-${r.relationship_type}">${r.relationship_type}</span>
          <span style="font-size: 12px; color: var(--text-muted);">Confidence: ${r.confidence}</span>
        </div>
        <p style="font-size: 12.5px; margin: 4px 0;">Interacts with: <strong>${escapeHtml(otherTitle)}</strong></p>
        <button class="btn btn-link btn-sm" onclick="openRelationshipModal('${r.relationship_id}')">View Cross-Clause Interaction</button>
      `;
      dom.modalRelatedList.appendChild(tag);
    });
  } else {
    dom.modalRelationshipsBlock.classList.add('hidden');
  }

  // Open Modal
  dom.clauseModal.classList.remove('hidden');
  dom.clauseModal.focus();

  // 4. Fetch / Display AI Explanation
  await fetchClauseExplanation(clause, false);
};

function closeClauseModal() {
  dom.clauseModal.classList.add('hidden');
  state.selectedClause = null;
}

async function fetchClauseExplanation(clause, forceRefresh = false) {
  // Check Cache
  if (!forceRefresh && state.clauseExplanationCache[clause.clause_id]) {
    renderClauseAIContent(state.clauseExplanationCache[clause.clause_id]);
    return;
  }

  dom.modalAILoader.classList.remove('hidden');
  dom.modalAIKeyPointsWrap.classList.add('hidden');
  dom.modalAIEvidenceWrap.classList.add('hidden');
  dom.modalAIUncertaintyWrap.classList.add('hidden');

  try {
    const res = await fetch(`${API_BASE_URL}/documents/${state.documentId}/explain-clause`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clause_id: clause.clause_id }),
    });

    if (!res.ok) {
      throw new Error(`API error: ${res.status}`);
    }

    const data = await res.json();
    state.clauseExplanationCache[clause.clause_id] = data;
    renderClauseAIContent(data);

  } catch (err) {
    console.warn('Failed to fetch clause explanation:', err);
    // Keep deterministic plain-English takeaway active so user is never left hanging
    if (!dom.modalAISummary.textContent) {
      dom.modalAISummary.textContent = (clause.attention_reasons && clause.attention_reasons.length > 0)
        ? `Quick Take: ${clause.attention_reasons.join('. ')}.`
        : 'Standard terms of service provision.';
    }
  } finally {
    dom.modalAILoader.classList.add('hidden');
  }
}

function renderClauseAIContent(data) {
  if (!data) return;

  if (data.status === 'llm_unavailable') {
    dom.modalAISummary.textContent = 'Gemini AI explanation is currently offline. Deterministic rule analysis and original contract evidence are displayed above.';
    return;
  }

  const exp = data.explanation;
  if (!exp) return;

  dom.modalAISummary.textContent = exp.summary || 'Summary unavailable.';
  dom.modalAIAttentionReason.textContent = exp.attention_explanation || 'Standard contractual terms noted.';

  // Key points
  if (exp.key_points && exp.key_points.length > 0) {
    dom.modalAIKeyPoints.innerHTML = '';
    exp.key_points.forEach(pt => {
      const li = document.createElement('li');
      li.textContent = pt;
      dom.modalAIKeyPoints.appendChild(li);
    });
    dom.modalAIKeyPointsWrap.classList.remove('hidden');
  }

  // Evidence Citations (Exact Verbatim Quotes)
  if (exp.evidence_references && exp.evidence_references.length > 0) {
    dom.modalAICitations.innerHTML = '';
    exp.evidence_references.forEach(ref => {
      const div = document.createElement('div');
      div.className = 'citation-item';
      div.innerHTML = `
        <span class="citation-loc">Citing: ${escapeHtml(ref.source_location)}</span>
        <em>"${escapeHtml(ref.quoted_text)}"</em>
      `;
      dom.modalAICitations.appendChild(div);
    });
    dom.modalAIEvidenceWrap.classList.remove('hidden');
  }

  // Insufficient evidence boundary
  if (!exp.evidence_sufficient || exp.uncertainty) {
    dom.modalAIUncertaintyWrap.classList.remove('hidden');
    dom.modalAIUncertainty.textContent = exp.uncertainty || 'Insufficient textual evidence to verify terms.';
  }
}

// ---------------------------------------------------------------------------
// Clause Relationships List & Modal Rendering
// ---------------------------------------------------------------------------
function getClauseDisplayTitle(clause, fallbackId) {
  if (!clause) return fallbackId || 'Clause';
  const num = clause.clause_index ? `Clause #${clause.clause_index}` : 'Clause';
  
  const hasRealTitle = clause.section_title && 
    clause.section_title.trim() && 
    !['general', 'pasted text', 'terms of service', 'unknown'].includes(clause.section_title.trim().toLowerCase());
    
  const cat = clause.primary_category || clause.category;
  const hasRealCat = cat && cat.trim() && !['general', 'unknown'].includes(cat.trim().toLowerCase());

  if (hasRealTitle && hasRealCat) {
    return `${num}: ${clause.section_title} (${cat})`;
  } else if (hasRealTitle) {
    return `${num}: ${clause.section_title}`;
  } else if (hasRealCat) {
    return `${num} (${cat})`;
  } else if (clause.text) {
    const clean = clause.text.replace(/\s+/g, ' ').trim();
    const snippet = clean.length > 28 ? clean.slice(0, 28) + '...' : clean;
    return `${num}: "${snippet}"`;
  }
  return num;
}

function renderRelationshipsList() {
  const container = dom.relationshipsContainer;
  container.innerHTML = '';

  if (state.relationships.length === 0) {
    container.innerHTML = `
      <div style="text-align: center; padding: 24px; color: var(--text-muted); font-size: 13.5px;">
        No cross-clause interactions or structural tensions detected in this agreement.
      </div>
    `;
    return;
  }

  state.relationships.forEach(rel => {
    const srcClause = state.clauses.find(c => c.clause_id === rel.source_clause_id);
    const tgtClause = state.clauses.find(c => c.clause_id === rel.target_clause_id);

    const srcTitle = getClauseDisplayTitle(srcClause, rel.source_clause_id);
    const tgtTitle = getClauseDisplayTitle(tgtClause, rel.target_clause_id);
    const arrowSymbol = rel.relationship_type === 'POTENTIAL_TENSION' ? '⇄' : '⟶';

    const triggerHtml = (rel.evidence && rel.evidence.trigger)
      ? `<div class="rel-trigger-row"><span class="rel-trigger-tag">Trigger</span> <span class="rel-trigger-quote">"${escapeHtml(rel.evidence.trigger)}"</span></div>`
      : '';

    const card = document.createElement('div');
    card.className = 'rel-card';
    card.setAttribute('role', 'listitem');

    card.innerHTML = `
      <div class="rel-card-header">
        <span class="rel-badge rel-badge-${rel.relationship_type}">${rel.relationship_type}</span>
        <span class="rel-confidence-pill">${(rel.confidence * 100).toFixed(0)}% confidence</span>
      </div>
      <div class="rel-nodes-preview">
        <span class="rel-node-chip src-chip" title="${escapeHtml(srcTitle)}">${escapeHtml(srcTitle)}</span>
        <span class="rel-arrow">${arrowSymbol}</span>
        <span class="rel-node-chip tgt-chip" title="${escapeHtml(tgtTitle)}">${escapeHtml(tgtTitle)}</span>
      </div>
      ${triggerHtml}
      <p class="rel-rationale-preview">${escapeHtml(rel.rationale)}</p>
      <div class="rel-card-footer">
        <span class="rel-loc-label">${srcClause && srcClause.source_location ? escapeHtml(srcClause.source_location.to_display_string || srcClause.source_location) : ''}</span>
        <button class="btn btn-secondary btn-sm" onclick="openRelationshipModal('${rel.relationship_id}')">
          Explain Interaction
        </button>
      </div>
    `;

    container.appendChild(card);
  });
}

window.openRelationshipModal = function(relId) {
  const rel = state.relationships.find(r => r.relationship_id === relId);
  if (!rel) return;

  state.selectedRelationship = rel;

  const srcClause = state.clauses.find(c => c.clause_id === rel.source_clause_id);
  const tgtClause = state.clauses.find(c => c.clause_id === rel.target_clause_id);

  const srcTitle = getClauseDisplayTitle(srcClause, rel.source_clause_id);
  const tgtTitle = getClauseDisplayTitle(tgtClause, rel.target_clause_id);
  const arrowSymbol = rel.relationship_type === 'POTENTIAL_TENSION' ? '⇄' : '⟶';

  dom.modalRelType.className = `rel-badge rel-badge-${rel.relationship_type}`;
  dom.modalRelType.textContent = rel.relationship_type;
  dom.modalRelHeadline.textContent = `${srcTitle} ${arrowSymbol} ${tgtTitle}`;
  dom.modalRelConfidence.textContent = `Algorithmic Confidence: ${(rel.confidence * 100).toFixed(0)}% (Rule-based heuristic)`;

  dom.relDiagSourceTitle.textContent = srcTitle;
  dom.relDiagSourceExcerpt.textContent = srcClause ? `"${srcClause.text.slice(0, 160)}..."` : '';

  dom.relDiagArrowLabel.className = `rel-type-chip rel-badge-${rel.relationship_type}`;
  dom.relDiagArrowLabel.textContent = rel.relationship_type;

  dom.relDiagTargetTitle.textContent = tgtTitle;
  dom.relDiagTargetExcerpt.textContent = tgtClause ? `"${tgtClause.text.slice(0, 160)}..."` : '';

  dom.modalRelRationale.textContent = rel.rationale;
  dom.modalRelTrigger.textContent = rel.evidence && rel.evidence.trigger ? rel.evidence.trigger : 'Cross-clause pattern match';

  // AI Content Reset
  dom.relAIContent.classList.add('hidden');
  dom.relAILoader.classList.add('hidden');
  dom.btnRequestRelAI.classList.remove('hidden');

  if (state.relationshipExplanationCache[rel.relationship_id]) {
    renderRelAIContent(state.relationshipExplanationCache[rel.relationship_id]);
  }

  dom.relModal.classList.remove('hidden');
  dom.relModal.focus();
};

function closeRelModal() {
  dom.relModal.classList.add('hidden');
  state.selectedRelationship = null;
}

async function fetchRelationshipExplanation(rel) {
  dom.relAILoader.classList.remove('hidden');
  dom.btnRequestRelAI.classList.add('hidden');

  try {
    const res = await fetch(`${API_BASE_URL}/documents/${state.documentId}/explain-relationship`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ relationship_id: rel.relationship_id }),
    });

    if (!res.ok) throw new Error(`API error: ${res.status}`);

    const data = await res.json();
    state.relationshipExplanationCache[rel.relationship_id] = data;
    renderRelAIContent(data);

  } catch (err) {
    console.error('Failed to explain relationship:', err);
    dom.modalRelAISummary.textContent = 'AI explanation could not be generated at this time.';
    dom.relAIContent.classList.remove('hidden');
  } finally {
    dom.relAILoader.classList.add('hidden');
  }
}

function renderRelAIContent(data) {
  if (!data) return;

  dom.btnRequestRelAI.classList.add('hidden');
  dom.relAIContent.classList.remove('hidden');

  if (data.status === 'llm_unavailable') {
    dom.modalRelAISummary.textContent = 'Gemini AI interaction analysis is offline. Deterministic rationale is presented above.';
    dom.modalRelAIEffect.textContent = '';
    return;
  }

  const exp = data.explanation;
  if (!exp) return;

  dom.modalRelAISummary.textContent = exp.summary || exp.relationship_explanation || 'Interaction explained.';
  dom.modalRelAIEffect.textContent = exp.attention_explanation || 'Consider how these provisions interact when executing rights or cancellations.';

  // Citations from both clauses
  if (exp.evidence_references && exp.evidence_references.length > 0) {
    dom.modalRelAICitations.innerHTML = '';
    exp.evidence_references.forEach(ref => {
      const div = document.createElement('div');
      div.className = 'citation-item';
      div.innerHTML = `
        <span class="citation-loc">[Clause ${escapeHtml(ref.clause_id)}] &bull; ${escapeHtml(ref.source_location)}</span>
        <em>"${escapeHtml(ref.quoted_text)}"</em>
      `;
      dom.modalRelAICitations.appendChild(div);
    });
  }
}

// ---------------------------------------------------------------------------
// UI Utilities & Helpers
// ---------------------------------------------------------------------------
function getAttentionCardClass(level) {
  switch (level) {
    case 'High Attention': return 'clause-card-high';
    case 'Medium Attention': return 'clause-card-medium';
    case 'Low Attention': return 'clause-card-low';
    default: return 'clause-card-info';
  }
}

function getAttentionBadgeClass(level) {
  switch (level) {
    case 'High Attention': return 'att-badge-high';
    case 'Medium Attention': return 'att-badge-medium';
    case 'Low Attention': return 'att-badge-low';
    default: return 'att-badge-info';
  }
}

function getAttentionBadgeHtml(level) {
  const cls = getAttentionBadgeClass(level);
  const inner = getAttentionBadgeInner(level);
  return `<span class="attention-badge ${cls}">${inner}</span>`;
}

function getAttentionBadgeInner(level) {
  switch (level) {
    case 'High Attention': return '⚠ High Attention';
    case 'Medium Attention': return '• Medium Attention';
    case 'Low Attention': return '• Low Attention';
    default: return 'ℹ Informational';
  }
}

function showAlert(title, message, type = 'danger') {
  dom.alertTitle.textContent = title;
  dom.alertMessage.textContent = message;
  dom.globalAlert.className = `alert alert-${type}`;
  dom.globalAlert.classList.remove('hidden');
}

function hideAlert() {
  dom.globalAlert.classList.add('hidden');
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
