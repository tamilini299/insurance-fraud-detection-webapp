// ================== CONFIG ==================
const API_BASE_URL = 'http://127.0.0.1:8000'; // FastAPI backend

// ================== HELPERS ==================
const $ = (sel) => document.querySelector(sel);
const val = (sel) => $(sel)?.value?.trim() ?? '';

function showToast(msg) {
  $('#toastText').textContent = msg || 'Done';
  $('#toast').classList.remove('hidden');
  setTimeout(() => $('#toast').classList.add('hidden'), 1800);
}

function setLoading(btn, isLoading, labelHTML) {
  if (!btn) return;
  if (isLoading) {
    btn.dataset.prev = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = labelHTML || '<i class="fa-solid fa-spinner fa-spin"></i>';
  } else {
    btn.disabled = false;
    btn.innerHTML = btn.dataset.prev || btn.innerHTML;
  }
}

function renderMessage({ role, html }) {
  const feed = $('#assistantOutput');
  const wrapper = document.createElement('div');
  wrapper.className = `msg ${role === 'user' ? 'user' : 'assistant'}`;
  wrapper.innerHTML = `
    <div class="avatar">${role === 'user' ? 'U' : 'AI'}</div>
    <div class="bubble">${html}</div>
  `;
  feed.appendChild(wrapper);
  // scroll to bottom
  const box = $('#assistantBox');
  if (box) box.scrollTop = box.scrollHeight;
}

// Auto-resize textarea
(function autoResize(){
  const ta = $('#assistantInput');
  if (!ta) return;
  const resize = () => {
    ta.style.height = 'auto';
    ta.style.height = Math.min(180, ta.scrollHeight) + 'px';
  };
  ta.addEventListener('input', resize);
  resize();
})();

// ================== LOGOUT ==================
$('#logout-btn')?.addEventListener('click', (e) => {
  e.preventDefault();
  window.location.href = 'index.html';
});

// ================== PREDICT ==================
$('#analyzeBtn')?.addEventListener('click', async () => {
  const providerId = val('#providerId');
  if (!providerId) return showToast('Enter Provider ID');

  const btn = $('#analyzeBtn');
  setLoading(btn, true, '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...');

  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider_id: providerId }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Server error');
    }
    const data = await res.json();
    const { riskScore, riskLevel, totalClaims, flaggedClaims, providerId: pid } = data;

    // Update UI
    $('#riskScore').textContent = `${riskScore}%`;
    $('#flaggedClaims').textContent = flaggedClaims;
    $('#totalClaims').textContent = totalClaims;
    $('#assessMeta').textContent = `Risk Assessment: ${pid} • Last analyzed: ${new Date().toLocaleString()}`;
    $('#riskBadge').textContent = riskLevel;

    $('#riskResult').classList.remove('hidden');

    // Prep assistant box
    $('#assistantBox').classList.remove('hidden');
    $('#assistantOutput').innerHTML = ''; // clear previous thread
    renderMessage({
      role: 'assistant',
      html: `<strong>Ready.</strong> Ask me anything about the analysis for <code>${pid}</code>.`,
    });

    showToast('Fraud risk analysis completed');
  } catch (e) {
    $('#riskResult').classList.add('hidden');
    showToast(`Error: ${e.message}`);
  } finally {
    setLoading(btn, false);
  }
});

// ================== EXPLAIN ==================
async function sendAssistantQuestion() {
  const q = val('#assistantInput');
  if (!q) return showToast('Ask a question first');

  // Must have results visible
  if ($('#riskResult').classList.contains('hidden')) {
    return showToast('Analyze a Provider ID first');
  }

  // Extract provider ID safely from meta
  const metaText = $('#assessMeta').textContent || '';
  const idMatch = metaText.match(/Risk Assessment:\s*([^\s•]+)/);
  const providerId = idMatch ? idMatch[1] : '';
  // Pull current metrics
  const riskScore = parseInt($('#riskScore').textContent) || 0;
  const flaggedClaims = parseInt($('#flaggedClaims').textContent) || 0;
  const totalClaims = parseInt($('#totalClaims').textContent) || 0;
  const riskLevel = $('#riskBadge').textContent || '';

  // Render user message
  renderMessage({ role: 'user', html: marked.parseInline(q) });
  $('#assistantInput').value = '';

  const btn = $('#assistantSend');
  setLoading(btn, true, '<i class="fa-solid fa-spinner fa-spin"></i>');

  try {
    const res = await fetch(`${API_BASE_URL}/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        provider_id: providerId,
        risk_score: riskScore,
        flagged_claims: flaggedClaims,
        total_claims: totalClaims,
        risk_level: riskLevel,
        user_question: q,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Explanation failed');
    }
    const data = await res.json();
    const html = window.marked ? marked.parse(data.explanation || '') : (data.explanation || '');
    renderMessage({ role: 'assistant', html });
    showToast('Assistant responded');
  } catch (e) {
    renderMessage({ role: 'assistant', html: `<em>Sorry, I couldn't generate an answer:</em> ${e.message}` });
    showToast(`Error: ${e.message}`);
  } finally {
    setLoading(btn, false, '<i class="fa-solid fa-paper-plane"></i>');
  }
}

$('#assistantSend')?.addEventListener('click', sendAssistantQuestion);
// submit on Enter (without Shift)
$('#assistantInput')?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendAssistantQuestion();
  }
});

// Clear & Copy actions
$('#assistantClear')?.addEventListener('click', () => {
  $('#assistantOutput').innerHTML = '';
  showToast('Cleared');
});
$('#assistantCopy')?.addEventListener('click', async () => {
  const text = $('#assistantOutput')?.innerText || '';
  try {
    await navigator.clipboard.writeText(text);
    showToast('Copied to clipboard');
  } catch {
    showToast('Copy failed');
  }
});
