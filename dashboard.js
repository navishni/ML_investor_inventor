const dashboardRoot = document.getElementById("main");
const dashboardRole = dashboardRoot?.dataset.role || "";
const dashboardEntityId = dashboardRoot?.dataset.entityId || "";
const bootstrap = window.MATCHTANK_BOOTSTRAP || {};
const context = window.MATCHTANK_CONTEXT || {};
const initialRecommendations = window.MATCHTANK_RECOMMENDATIONS || [];
const initialMatches = window.MATCHTANK_MATCHES || [];
const initialConversations = context.conversations || [];
const initialChatDirectory = context.chatContacts || [];

let allInvestorRecommendations = [];
let currentInvestorRecommendations = [];
let allInventorMatches = [];
let currentInventorMatches = [];
let inboxConversations = initialConversations.slice();
let allChatDirectory = initialChatDirectory.slice();
let chatThreadSearch = "";
let chatSearchTimer = null;
let activeChat = null;
let chatRefreshTimer = null;
let inboxRefreshTimer = null;
let feedbackState = context.feedback || {};

if (initialRecommendations.length) {
  allInvestorRecommendations = initialRecommendations;
  currentInvestorRecommendations = initialRecommendations;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return response.json();
}

function getShortlist() {
  return JSON.parse(localStorage.getItem("matchtank_shortlist") || "[]");
}

function saveShortlist(items) {
  localStorage.setItem("matchtank_shortlist", JSON.stringify(items));
}

function renderShortlist() {
  const panel = document.getElementById("shortlistPanel");
  if (!panel) return;
  const shortlist = getShortlist();
  panel.innerHTML = shortlist.length
    ? `<strong>Shortlist</strong>: ${shortlist.map((item) => item.title).join(", ")}`
    : "Shortlist is empty.";
}

function setPanelVisible(panelId, visible) {
  const panel = document.getElementById(panelId);
  if (!panel) return;
  panel.hidden = !visible;
}

function formatTimestamp(timestamp) {
  if (!timestamp) return "";
  try {
    return new Date(timestamp).toLocaleString();
  } catch {
    return timestamp;
  }
}

function updateUnreadBadge(count) {
  const badge = document.getElementById("chatUnreadBadge");
  if (!badge) return;
  const value = Number(count) || 0;
  badge.textContent = value > 99 ? "99+" : String(value);
  badge.hidden = value <= 0;
}

function normalizeSearchText(value) {
  return String(value || "").toLowerCase();
}

function buildSearchBlob(value) {
  if (Array.isArray(value)) {
    return value.map(buildSearchBlob).join(" ");
  }
  if (value && typeof value === "object") {
    return Object.values(value).map(buildSearchBlob).join(" ");
  }
  return normalizeSearchText(value);
}

function filterChatThreads(conversations) {
  const query = chatThreadSearch.trim().toLowerCase();
  if (!query) {
    return conversations;
  }
  return conversations.filter((thread) => {
    const haystack = buildSearchBlob(thread);
    return haystack.includes(query);
  });
}

function getConversationKey(investorId, ideaId) {
  return `${Number(investorId)}:${Number(ideaId)}`;
}

function getDirectoryConversationKey(item) {
  if (dashboardRole === "investor") {
    return getConversationKey(dashboardEntityId, item.idea_id);
  }
  return getConversationKey(item.investor_id, dashboardEntityId);
}

function filterChatDirectory(items) {
  const query = chatThreadSearch.trim().toLowerCase();
  const conversationKeys = new Set(inboxConversations.map((thread) => getConversationKey(thread.investor_id, thread.idea_id)));
  return items.filter((item) => {
    const key = getDirectoryConversationKey(item);
    if (conversationKeys.has(key)) {
      return false;
    }
    if (!query) {
      return true;
    }
    const haystack = buildSearchBlob(item);
    return haystack.includes(query);
  });
}

async function refreshInboxState() {
  try {
    const response = await fetch("/api/chat/inbox");
    if (!response.ok) return;
    const data = await response.json();
    updateUnreadBadge(data.unread_count || 0);
    inboxConversations = data.conversations || [];
    renderChatThreadList(inboxConversations);
    renderChatDirectory(allChatDirectory);
    if (activeChat) {
      highlightChatThread(getChatKey(activeChat));
    }
  } catch {
    return;
  }
}

function scrollToChatPanel() {
  const panel = document.getElementById("chatPanel");
  if (panel) {
    panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderChatThreadList(conversations) {
  inboxConversations = Array.isArray(conversations) ? conversations : [];
  const threadList = document.getElementById("chatConversationList");
  if (!threadList) return;
  const visibleConversations = filterChatThreads(inboxConversations);
  if (!visibleConversations.length) {
    threadList.innerHTML = chatThreadSearch.trim()
      ? '<p class="note">No conversations match this search.</p>'
      : '<p class="note">No conversations yet. Open a recommendation to start one.</p>';
    return;
  }
  threadList.innerHTML = visibleConversations.map((thread) => `
    <button
      type="button"
      class="secondary-btn chat-thread-btn${thread.has_unread ? ' has-unread' : ''}"
      data-investor-id="${thread.investor_id}"
      data-idea-id="${thread.idea_id}"
      data-label="${thread.label}"
      data-counterparty="${thread.counterparty_name}"
    >
      <strong>${thread.label}</strong>
      <span>${thread.counterparty_name}</span>
      <small>Last by ${thread.last_sender || thread.counterparty_name}</small>
      <small>${thread.last_message_preview || 'No messages yet.'}</small>
      ${thread.unread_count ? `<span class="badge-count thread-badge">${thread.unread_count}</span>` : ''}
    </button>
  `).join("");
}

function renderChatDirectory(items) {
  allChatDirectory = Array.isArray(items) ? items : [];
  const directoryList = document.getElementById("chatDirectoryList");
  if (!directoryList) return;
  const visibleItems = filterChatDirectory(allChatDirectory);
  if (!visibleItems.length) {
    directoryList.innerHTML = chatThreadSearch.trim()
      ? '<p class="note">No people match this search.</p>'
      : '<p class="note">No suggested people available yet.</p>';
    return;
  }
  directoryList.innerHTML = visibleItems.map((item) => {
    const displayName = item.display_name || item.idea_title || item.investor_name || "Contact";
    const counterpartyName = item.counterparty_name || displayName;
    const subtitle = item.subtitle || "";
    const detailLine = item.detail_line || "";
    const summary = item.summary || "";
    const investorId = item.investor_id ?? dashboardEntityId;
    const ideaId = item.idea_id ?? dashboardEntityId;
    return `
      <button
        type="button"
        class="secondary-btn chat-open-btn chat-directory-btn"
        data-investor-id="${investorId}"
        data-idea-id="${ideaId}"
        data-label="${displayName}"
        data-counterparty="${counterpartyName}"
      >
        <strong>${displayName}</strong>
        <span>${subtitle}</span>
        <small>${detailLine}</small>
        <small>${summary}</small>
      </button>
    `;
  }).join("");
}

async function fetchChatContacts(query = "") {
  const response = await fetch(`/api/chat/contacts?q=${encodeURIComponent(query)}&limit=100`);
  if (!response.ok) {
    return;
  }
  const data = await response.json();
  allChatDirectory = data.contacts || [];
  renderChatDirectory(allChatDirectory);
}

function renderChatMessages(messages) {
  const chatMessages = document.getElementById("chatMessages");
  if (!chatMessages) return;
  chatMessages.innerHTML = messages.length
    ? messages.map((message) => {
        const isSelf = message.sender_role === dashboardRole;
        return `
          <article class="chat-message ${isSelf ? "self" : ""}">
            <div class="meta-line">${message.sender_name} &middot; ${formatTimestamp(message.timestamp)}</div>
            <div>${message.message}</div>
          </article>
        `;
      }).join("")
    : `<div class="note">No messages yet. Start the conversation below.</div>`;
}

async function markCurrentThreadRead() {
  if (!activeChat) return;
  await fetch(`/api/chat/${activeChat.investorId}/${activeChat.ideaId}/read`, { method: "POST" });
  await refreshInboxState();
}

async function loadChatThread(investorId, ideaId, label, counterpartyName = "") {
  if (!investorId || !ideaId) return;
  activeChat = {
    investorId: Number(investorId),
    ideaId: Number(ideaId),
    label: label || "Match",
    counterpartyName,
  };
  setChatMeta(`Thread with ${activeChat.label}${activeChat.counterpartyName ? ` | ${activeChat.counterpartyName}` : ""}.`);
  highlightChatThread(getChatKey(activeChat));
  scrollToChatPanel();
  const response = await fetch(`/api/chat/${activeChat.investorId}/${activeChat.ideaId}`);
  const data = await response.json();
  renderChatMessages(data.messages || []);
  await markCurrentThreadRead();
  startChatPolling();
}

async function loadInventorDetail(ideaId) {
  const response = await fetch(`/api/inventor/${ideaId}/detail`);
  const data = await response.json();
  const detailContent = document.getElementById("detailContent");
  if (!detailContent) return;
  detailContent.innerHTML = `
    <div class="grid stats-grid" style="grid-template-columns: repeat(3, minmax(0, 1fr));">
      <article class="card stat-card"><span>Project</span><strong>${data.idea_title || ""}</strong></article>
      <article class="card stat-card"><span>Founder</span><strong>${data.founder_name || ""}</strong></article>
      <article class="card stat-card"><span>Domain</span><strong>${data.domain || ""}</strong></article>
    </div>
    <p class="lead small">${data.description || data.idea_text || ""}</p>
    <div class="card" style="padding: 1rem; margin-top: 1rem;">
      <p class="eyebrow">Achievements</p>
      <p>${data.achievements || "No achievements added yet."}</p>
    </div>
    <div class="card" style="padding: 1rem; margin-top: 1rem;">
      <p class="eyebrow">Patents</p>
      <p>${data.patents || "No patents added yet."}</p>
    </div>
  `;
  setPanelVisible("detailPanel", true);
}

async function loadInvestorDetail(investorId) {
  const response = await fetch(`/api/investor/${investorId}/detail`);
  const data = await response.json();
  const detailContent = document.getElementById("investorDetailContent");
  if (!detailContent) return;
  detailContent.innerHTML = `
    <div class="grid stats-grid" style="grid-template-columns: repeat(3, minmax(0, 1fr));">
      <article class="card stat-card"><span>Investor</span><strong>${data.investor_name || ""}</strong></article>
      <article class="card stat-card"><span>Focus</span><strong>${data.focus_domain || ""}</strong></article>
      <article class="card stat-card"><span>Risk</span><strong>${data.preferred_risk_appetite || ""}</strong></article>
    </div>
    <p class="lead small">${data.industry_focus || ""}</p>
    <div class="card" style="padding: 1rem; margin-top: 1rem;">
      <p class="eyebrow">Location</p>
      <p>${data.preferred_location || "Not set"}</p>
    </div>
    <div class="card" style="padding: 1rem; margin-top: 1rem;">
      <p class="eyebrow">Funding profile</p>
      <p>Available funds: ${Number(data.available_funds || 0).toLocaleString()}</p>
      <p>Company investment: ${Number(data.company_investment || 0).toLocaleString()}</p>
      <p>Past investments: ${Number(data.past_investments || 0).toLocaleString()}</p>
    </div>
  `;
  setPanelVisible("investorDetailPanel", true);
}

async function saveFeedback(ideaId, decision) {
  const response = await postJson("/api/recommendation-feedback", {
    idea_id: Number(ideaId),
    decision,
  });
  feedbackState[String(ideaId)] = decision;
  const statusText = decision === "like" ? "Liked" : "Disliked";
  renderInvestorRecommendations(document.getElementById("investorRecommendations"), currentInvestorRecommendations);
  const shortlistPanel = document.getElementById("shortlistPanel");
  if (shortlistPanel) {
    shortlistPanel.textContent = `${statusText} idea ${ideaId}. ${response.message || ""}`;
  }
}

async function sendChatMessage(event) {
  event.preventDefault();
  if (!activeChat) return;
  const input = document.getElementById("chatInput");
  const message = input?.value.trim();
  if (!message) return;
  const response = await postJson(`/api/chat/${activeChat.investorId}/${activeChat.ideaId}`, { message });
  input.value = "";
  renderChatMessages(response.messages || []);
  await markCurrentThreadRead();
}

function renderInvestorRecommendations(container, items) {
  currentInvestorRecommendations = items;
  if (!allInvestorRecommendations.length || items.length >= allInvestorRecommendations.length) {
    allInvestorRecommendations = items;
  }
  container.innerHTML = items.map((item) => {
    const reaction = feedbackState[String(item.idea_id)] || "";
    return `
      <article class="recommendation-card">
        <div class="recommendation-head">
          <div>
            <h3>${item.idea_title}</h3>
            <p>${item.domain} | ${item.technology} | ${item.risk_level} risk | ${item.location}</p>
            <p>Funding required: ${Number(item.funding_required).toLocaleString()} | Team size: ${item.team_size}</p>
          </div>
          <div class="score-box"><strong>${item.best_score}%</strong><span>${item.best_model}</span></div>
        </div>
        <p>${item.summary}</p>
        <div class="chip-row">
          ${item.reasons.map((reason) => `<span class="chip">${reason}</span>`).join("")}
          ${reaction ? `<span class="chip">${reaction === "like" ? "Liked" : "Disliked"}</span>` : ""}
        </div>
        <div class="top-actions" style="margin-top:0.75rem; flex-wrap: wrap;">
          <button type="button" class="secondary-btn shortlist-btn" data-id="${item.idea_id}" data-title="${item.idea_title}">Add to shortlist</button>
          <button type="button" class="secondary-btn feedback-btn" data-action="like" data-idea-id="${item.idea_id}">Like</button>
          <button type="button" class="secondary-btn feedback-btn" data-action="dislike" data-idea-id="${item.idea_id}">Dislike</button>
          <button type="button" class="secondary-btn detail-btn" data-idea-id="${item.idea_id}">View achievements</button>
          <button type="button" class="secondary-btn chat-btn" data-action="chat" data-idea-id="${item.idea_id}" data-investor-id="${dashboardEntityId}" data-label="${item.idea_title}" data-counterparty="Inventor">Chat</button>
        </div>
      </article>
    `;
  }).join("");
  renderShortlist();
}

function renderInventorMatches(container, items) {
  currentInventorMatches = items;
  if (!allInventorMatches.length || items.length >= allInventorMatches.length) {
    allInventorMatches = items;
  }
  container.innerHTML = items.map((item) => `
    <article class="recommendation-card">
      <div class="recommendation-head">
        <div>
          <h3>${item.investor_name}</h3>
          <p>${item.focus_domain} | ${item.industry_focus}</p>
          <p>${item.preferred_location} | ${item.preferred_risk_appetite}</p>
        </div>
        <div class="score-box"><strong>${item.best_score}%</strong><span>${item.best_model}</span></div>
        </div>
        <div class="chip-row">${item.reasons.map((reason) => `<span class="chip">${reason}</span>`).join("")}</div>
        <div class="top-actions" style="margin-top:0.75rem; flex-wrap: wrap;">
        <button type="button" class="secondary-btn detail-btn-investor" data-investor-id="${item.investor_id}">View investor</button>
        <button type="button" class="secondary-btn chat-btn" data-action="chat" data-investor-id="${item.investor_id}" data-idea-id="${dashboardEntityId}" data-label="${item.investor_name}" data-counterparty="${item.investor_name}">Chat</button>
      </div>
    </article>
  `).join("");
}

function setupFilters() {
  const domainInput = document.getElementById("domainFilter");
  const riskInput = document.getElementById("riskFilter");
  const searchInput = document.getElementById("searchFilter");
  const container = document.getElementById("investorRecommendations");
  if (!domainInput || !riskInput || !searchInput || !container) return;

  const applyFilters = () => {
    const domainValue = domainInput.value.trim().toLowerCase();
    const riskValue = riskInput.value.trim().toLowerCase();
    const searchValue = searchInput.value.trim().toLowerCase();
    const filtered = allInvestorRecommendations.filter((item) => {
      const domainMatch = !domainValue || item.domain.toLowerCase().includes(domainValue);
      const riskMatch = !riskValue || item.risk_level.toLowerCase().includes(riskValue);
      const text = `${item.idea_title} ${item.technology} ${item.summary}`.toLowerCase();
      const searchMatch = !searchValue || text.includes(searchValue);
      return domainMatch && riskMatch && searchMatch;
    });
    renderInvestorRecommendations(container, filtered);
  };

  domainInput.addEventListener("input", applyFilters);
  riskInput.addEventListener("input", applyFilters);
  searchInput.addEventListener("input", applyFilters);
}

function setupInventorFilters() {
  const searchInput = document.getElementById("investorSearchFilter");
  const domainInput = document.getElementById("investorDomainFilter");
  const locationInput = document.getElementById("investorLocationFilter");
  const container = document.getElementById("inventorMatches");
  if (!searchInput || !domainInput || !locationInput || !container) return;

  const applyFilters = () => {
    const searchValue = searchInput.value.trim().toLowerCase();
    const domainValue = domainInput.value.trim().toLowerCase();
    const locationValue = locationInput.value.trim().toLowerCase();
    const filtered = allInventorMatches.filter((item) => {
      const text = `${item.investor_name} ${item.focus_domain} ${item.industry_focus} ${item.preferred_location} ${item.preferred_risk_appetite}`.toLowerCase();
      const searchMatch = !searchValue || text.includes(searchValue);
      const domainMatch = !domainValue || text.includes(domainValue);
      const locationMatch = !locationValue || item.preferred_location.toLowerCase().includes(locationValue);
      return searchMatch && domainMatch && locationMatch;
    });
    renderInventorMatches(container, filtered);
  };

  searchInput.addEventListener("input", applyFilters);
  domainInput.addEventListener("input", applyFilters);
  locationInput.addEventListener("input", applyFilters);
}

function setupChatThreadSearch() {
  const input = document.getElementById("chatThreadSearch");
  if (!input) return;
  chatThreadSearch = input.value || "";
  input.addEventListener("input", () => {
    chatThreadSearch = input.value || "";
    renderChatThreadList(inboxConversations);
    if (chatSearchTimer) {
      clearTimeout(chatSearchTimer);
    }
    chatSearchTimer = setTimeout(() => {
      void fetchChatContacts(chatThreadSearch);
    }, 220);
    if (activeChat) {
      highlightChatThread(getChatKey(activeChat));
    }
  });
}

function bindContainers() {
  const investorContainer = document.getElementById("investorRecommendations");
  const inventorContainer = document.getElementById("inventorMatches");
  const chatThreadList = document.getElementById("chatSidebar");

  if (chatThreadList) {
    chatThreadList.addEventListener("click", async (event) => {
      const button = event.target.closest("button");
      if (!button || !button.classList.contains("chat-open-btn")) return;
      await loadChatThread(
        button.dataset.investorId,
        button.dataset.ideaId,
        button.dataset.label || "Match",
        button.dataset.counterparty || ""
      );
    });
  }

  if (investorContainer) {
    investorContainer.addEventListener("click", async (event) => {
      const button = event.target.closest("button");
      if (!button) return;

      const action = button.dataset.action;
      if (button.classList.contains("shortlist-btn")) {
        const shortlist = getShortlist();
        const next = { id: button.dataset.id, title: button.dataset.title };
        if (!shortlist.find((item) => item.id === next.id)) {
          shortlist.push(next);
          saveShortlist(shortlist);
        }
        renderShortlist();
        return;
      }

      if (action === "like" || action === "dislike") {
        await saveFeedback(button.dataset.ideaId, action);
        return;
      }

      if (button.classList.contains("detail-btn")) {
        await loadInventorDetail(button.dataset.ideaId);
        return;
      }

      if (action === "chat" || button.classList.contains("chat-btn")) {
        await loadChatThread(button.dataset.investorId || dashboardEntityId, button.dataset.ideaId, button.dataset.label || "Inventor");
      }
    });
  }

  if (inventorContainer) {
    inventorContainer.addEventListener("click", async (event) => {
      const button = event.target.closest("button");
      if (!button) return;
      if (button.classList.contains("detail-btn-investor")) {
        await loadInvestorDetail(button.dataset.investorId);
        return;
      }
      if (button.classList.contains("chat-btn")) {
        await loadChatThread(
          button.dataset.investorId,
          button.dataset.ideaId,
          button.dataset.label || "Investor",
          button.dataset.counterparty || ""
        );
      }
    });
  }
}

const equityForm = document.getElementById("equityForm");
if (equityForm) {
  equityForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const result = document.getElementById("equityResult");
    const payload = {
      pre_money_valuation: document.getElementById("preMoney").value,
      investment_amount: document.getElementById("investmentAmount").value,
      founder_equity_before: document.getElementById("founderEquityBefore").value,
    };
    const data = await postJson("/api/equity-calculator", payload);
    result.textContent = data.error
      ? data.error
      : `Post-money valuation: ${Number(data.post_money_valuation).toLocaleString()} | Investor equity: ${data.investor_equity_percent}% | Founder equity after deal: ${data.founder_equity_after_percent}%`;
  });
}

const refreshInvestorBtn = document.getElementById("refreshInvestorBtn");
if (refreshInvestorBtn) {
  refreshInvestorBtn.addEventListener("click", async () => {
    const investorId = refreshInvestorBtn.dataset.investorId;
    const response = await fetch(`/api/investor/${investorId}/recommendations`);
    const data = await response.json();
    allInvestorRecommendations = data.recommendations || [];
    currentInvestorRecommendations = allInvestorRecommendations;
    renderInvestorRecommendations(document.getElementById("investorRecommendations"), allInvestorRecommendations);
  });
}

const refreshInventorBtn = document.getElementById("refreshInventorBtn");
if (refreshInventorBtn) {
  refreshInventorBtn.addEventListener("click", async () => {
    const ideaId = refreshInventorBtn.dataset.ideaId;
    const response = await fetch(`/api/inventor/${ideaId}/matches`);
    const data = await response.json();
    renderInventorMatches(document.getElementById("inventorMatches"), data.matches || []);
  });
}

const viewShortlistBtn = document.getElementById("viewShortlistBtn");
if (viewShortlistBtn) {
  viewShortlistBtn.addEventListener("click", renderShortlist);
}

const profileForm = document.getElementById("inventorProfileForm");
if (profileForm) {
  profileForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const ideaId = profileForm.dataset.ideaId;
    const status = document.getElementById("profileStatus");
    const data = await postJson(`/api/inventor/${ideaId}/profile`, {
      founder_name: document.getElementById("founderName").value,
      email: document.getElementById("email").value,
      linkedin: document.getElementById("linkedin").value,
      description: document.getElementById("description").value,
      achievements: document.getElementById("achievements").value,
      patents: document.getElementById("patents").value,
    });
    status.textContent = data.error || data.message;
  });
}

const chatForm = document.getElementById("chatForm");
if (chatForm) {
  chatForm.addEventListener("submit", sendChatMessage);
}

const closeDetailBtn = document.getElementById("closeDetailBtn");
if (closeDetailBtn) {
  closeDetailBtn.addEventListener("click", () => setPanelVisible("detailPanel", false));
}

const closeInvestorDetailBtn = document.getElementById("closeInvestorDetailBtn");
if (closeInvestorDetailBtn) {
  closeInvestorDetailBtn.addEventListener("click", () => setPanelVisible("investorDetailPanel", false));
}

const closeChatBtn = document.getElementById("closeChatBtn");
if (closeChatBtn) {
  closeChatBtn.addEventListener("click", () => clearChatSelection());
}

const chatInboxBtn = document.getElementById("chatInboxBtn");
if (chatInboxBtn) {
  chatInboxBtn.addEventListener("click", async () => {
    scrollToChatPanel();
    if (activeChat) return;
    const response = await fetch("/api/chat/inbox");
    const data = await response.json();
    const threads = data.conversations || initialConversations;
    const firstThread = [...threads].find((thread) => thread.has_unread) || threads[0];
    if (firstThread) {
      await loadChatThread(
        firstThread.investor_id,
        firstThread.idea_id,
        firstThread.label,
        firstThread.counterparty_name
      );
    }
  });
}

function getChatKey(chat) {
  if (!chat) return "";
  return `${Number(chat.investorId)}:${Number(chat.ideaId)}`;
}

function highlightChatThread(activeThreadKey) {
  const buttons = document.querySelectorAll(".chat-thread-btn");
  buttons.forEach((button) => {
    const key = `${Number(button.dataset.investorId)}:${Number(button.dataset.ideaId)}`;
    button.classList.toggle("is-active", key === activeThreadKey);
  });
}

function setChatMeta(text) {
  const chatMeta = document.getElementById("chatMeta");
  if (chatMeta) {
    chatMeta.textContent = text;
  }
}

function renderChatMessages(messages) {
  const chatMessages = document.getElementById("chatMessages");
  if (!chatMessages) return;
  chatMessages.innerHTML = messages.length
    ? messages.map((message) => {
        const isSelf = message.sender_role === dashboardRole;
        return `
          <article class="chat-message ${isSelf ? "self" : "incoming"}">
            <div class="meta-line">${message.sender_name} &middot; ${formatTimestamp(message.timestamp)}</div>
            <div class="message-body">${message.message}</div>
          </article>
        `;
      }).join("")
    : `<div class="note">No messages yet. Start the conversation below.</div>`;
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function stopChatPolling() {
  if (chatRefreshTimer) {
    clearInterval(chatRefreshTimer);
    chatRefreshTimer = null;
  }
}

async function refreshActiveChat() {
  if (!activeChat) return;
  const response = await fetch(`/api/chat/${activeChat.investorId}/${activeChat.ideaId}`);
  const data = await response.json();
  renderChatMessages(data.messages || []);
  await markCurrentThreadRead();
}

function startChatPolling() {
  stopChatPolling();
  if (!activeChat) return;
  chatRefreshTimer = setInterval(() => {
    void refreshActiveChat();
  }, 3500);
}

async function loadChatThread(investorId, ideaId, label, counterpartyName = "") {
  if (!investorId || !ideaId) return;
  activeChat = {
    investorId: Number(investorId),
    ideaId: Number(ideaId),
    label: label || "Match",
    counterpartyName,
  };
  setChatMeta(`Thread with ${activeChat.label}${activeChat.counterpartyName ? ` | ${activeChat.counterpartyName}` : ""}.`);
  highlightChatThread(getChatKey(activeChat));
  const response = await fetch(`/api/chat/${activeChat.investorId}/${activeChat.ideaId}`);
  const data = await response.json();
  renderChatMessages(data.messages || []);
  startChatPolling();
}

function clearChatSelection() {
  stopChatPolling();
  activeChat = null;
  highlightChatThread("");
  setChatMeta("Select a conversation from the left or open someone from People to contact.");
  renderChatMessages([]);
}

document.addEventListener("DOMContentLoaded", () => {
  const investorContainer = document.getElementById("investorRecommendations");
  if (investorContainer && initialRecommendations.length) {
    renderInvestorRecommendations(investorContainer, initialRecommendations);
  }
  const inventorContainer = document.getElementById("inventorMatches");
  if (inventorContainer && initialMatches.length) {
    renderInventorMatches(inventorContainer, initialMatches);
  }
  bindContainers();
  renderShortlist();
  setupFilters();
  setupInventorFilters();
  setupChatThreadSearch();
  renderChatDirectory(allChatDirectory);
  void refreshInboxState();
  inboxRefreshTimer = setInterval(() => {
    void refreshInboxState();
  }, 5000);
  clearChatSelection();
});
