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
    resDocTitle: document.getElementById('resDocTitle'),
    btnReanalyze: document.getElementById('btnReanalyze'),
    synthesisSummary: document.getElementById('synthesisSummary'),
    synthesisKeyPoints: document.getElementById('synthesisKeyPoints'),
    synthesisDegradedNotice: document.getElementById('synthesisDegradedNotice'),

    // Stat Cards
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
  dom.btnReanalyze.addEventListener('click', resetToInputSection);

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
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Failed to upload and parse PDF document.');
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
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Failed to submit and segment text.');
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
