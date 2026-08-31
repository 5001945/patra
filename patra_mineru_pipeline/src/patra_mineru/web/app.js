import * as pdfjsLib from "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";

const state = {
  jobId: null,
  pdf: null,
  document: null,
  blocksById: new Map(),
  selectedBlock: null,
  zoom: 1,
  fileName: "PDF 없음",
};

const el = {
  dropZone: document.querySelector("#dropZone"),
  pdfInput: document.querySelector("#pdfInput"),
  statusPill: document.querySelector("#statusPill"),
  runParse: document.querySelector("#runParse"),
  mineruOutput: document.querySelector("#mineruOutput"),
  protectedTerms: document.querySelector("#protectedTerms"),
  taskInput: document.querySelector("#taskInput"),
  baseUrl: document.querySelector("#baseUrl"),
  modelName: document.querySelector("#modelName"),
  apiKey: document.querySelector("#apiKey"),
  jsonMode: document.querySelector("#jsonMode"),
  viewer: document.querySelector("#viewer"),
  docTitle: document.querySelector("#docTitle"),
  docMeta: document.querySelector("#docMeta"),
  zoomOut: document.querySelector("#zoomOut"),
  zoomIn: document.querySelector("#zoomIn"),
  zoomLabel: document.querySelector("#zoomLabel"),
  detailEmpty: document.querySelector("#detailEmpty"),
  detailContent: document.querySelector("#detailContent"),
  selectedId: document.querySelector("#selectedId"),
  cropImage: document.querySelector("#cropImage"),
  sourceText: document.querySelector("#sourceText"),
  translateBtn: document.querySelector("#translateBtn"),
  translationText: document.querySelector("#translationText"),
  closeDetail: document.querySelector("#closeDetail"),
};

function setStatus(status, message = status) {
  el.statusPill.textContent = message;
  el.statusPill.className = `status-pill ${status}`;
}

function saveSettings() {
  localStorage.setItem("patraPdfSettings", JSON.stringify({
    baseUrl: el.baseUrl.value,
    modelName: el.modelName.value,
    protectedTerms: el.protectedTerms.value,
    task: el.taskInput.value,
    jsonMode: el.jsonMode.checked,
  }));
}

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("patraPdfSettings") || "{}");
    if (saved.baseUrl) el.baseUrl.value = saved.baseUrl;
    if (saved.modelName) el.modelName.value = saved.modelName;
    if (saved.protectedTerms) el.protectedTerms.value = saved.protectedTerms;
    if (saved.task) el.taskInput.value = saved.task;
    if (typeof saved.jsonMode === "boolean") el.jsonMode.checked = saved.jsonMode;
  } catch {}
}

async function uploadPdf(file) {
  saveSettings();
  state.fileName = file.name;
  setStatus("busy", "uploading");
  el.docTitle.textContent = file.name;
  el.docMeta.textContent = "업로드 중";
  el.viewer.classList.remove("empty-state");
  el.viewer.innerHTML = "";

  const form = new FormData();
  form.append("pdf", file);
  form.append("task", el.taskInput.value);
  form.append("protected_terms", el.protectedTerms.value);
  form.append("run_parse", el.runParse.checked ? "true" : "false");
  if (el.mineruOutput.value.trim()) form.append("mineru_output", el.mineruOutput.value.trim());

  const res = await fetch("/api/jobs", { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  state.jobId = data.job_id;
  await pollJob();
}

async function pollJob() {
  for (;;) {
    const res = await fetch(`/api/jobs/${state.jobId}/status`);
    const status = await res.json();
    if (status.status === "ready") {
      setStatus("ready", "ready");
      await loadDocument();
      return;
    }
    if (status.status === "error") {
      setStatus("error", "error");
      el.docMeta.textContent = status.message || "처리 실패";
      throw new Error(status.message || "Job failed");
    }
    setStatus("busy", status.message || status.status);
    el.docMeta.textContent = status.message || status.status;
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

async function loadDocument() {
  const [docRes, pdf] = await Promise.all([
    fetch(`/api/jobs/${state.jobId}/document`).then((r) => r.json()),
    pdfjsLib.getDocument(`/api/jobs/${state.jobId}/pdf`).promise,
  ]);
  state.document = docRes;
  state.pdf = pdf;
  state.blocksById = new Map(docRes.blocks.map((block) => [block.id, block]));
  el.docTitle.textContent = state.fileName;
  el.docMeta.textContent = `${docRes.pages.length} pages · ${docRes.blocks.length} selectable blocks`;
  await renderPdf();
}

async function renderPdf() {
  if (!state.pdf || !state.document) return;
  el.viewer.innerHTML = "";
  el.zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;

  for (let pageNumber = 1; pageNumber <= state.pdf.numPages; pageNumber += 1) {
    const page = await state.pdf.getPage(pageNumber);
    const baseViewport = page.getViewport({ scale: 1 });
    const maxWidth = Math.max(420, el.viewer.clientWidth - 48);
    const fitScale = Math.min(1.55, maxWidth / baseViewport.width);
    const viewport = page.getViewport({ scale: fitScale * state.zoom });

    const pageWrap = document.createElement("section");
    pageWrap.className = "pdf-page";
    pageWrap.style.width = `${viewport.width}px`;
    pageWrap.style.height = `${viewport.height}px`;

    const canvas = document.createElement("canvas");
    canvas.width = Math.floor(viewport.width * window.devicePixelRatio);
    canvas.height = Math.floor(viewport.height * window.devicePixelRatio);
    canvas.style.width = `${viewport.width}px`;
    canvas.style.height = `${viewport.height}px`;
    const context = canvas.getContext("2d");
    context.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    await page.render({ canvasContext: context, viewport }).promise;

    const overlay = document.createElement("div");
    overlay.className = "overlay-layer";
    overlay.style.width = `${viewport.width}px`;
    overlay.style.height = `${viewport.height}px`;

    const pageBlocks = state.document.blocks.filter((block) => block.page_idx === pageNumber - 1);
    for (const block of pageBlocks) {
      overlay.appendChild(makeBox(block, viewport));
    }

    pageWrap.append(canvas, overlay);
    el.viewer.appendChild(pageWrap);
  }
}

function makeBox(block, viewport) {
  const [x0, y0, x1, y1] = block.bbox;
  const box = document.createElement("button");
  box.className = "block-box";
  box.type = "button";
  box.dataset.blockId = block.id;
  box.style.left = `${(x0 / 1000) * viewport.width}px`;
  box.style.top = `${(y0 / 1000) * viewport.height}px`;
  box.style.width = `${((x1 - x0) / 1000) * viewport.width}px`;
  box.style.height = `${((y1 - y0) / 1000) * viewport.height}px`;
  box.title = `${block.id}: ${block.text.slice(0, 120)}`;
  box.innerHTML = `<span>${block.text_level ? "section" : block.id}</span>`;
  box.addEventListener("click", () => selectBlock(block.id));
  return box;
}

async function selectBlock(blockId) {
  state.selectedBlock = state.blocksById.get(blockId);
  document.querySelectorAll(".block-box.selected").forEach((node) => node.classList.remove("selected"));
  document.querySelector(`[data-block-id="${CSS.escape(blockId)}"]`)?.classList.add("selected");

  el.detailEmpty.classList.add("hidden");
  el.detailContent.classList.remove("hidden");
  el.selectedId.textContent = blockId;
  el.sourceText.textContent = state.selectedBlock.text;
  el.translationText.textContent = "";
  el.cropImage.src = `/api/jobs/${state.jobId}/blocks/${blockId}/crop.png`;

  try {
    const cached = await fetch(`/api/jobs/${state.jobId}/blocks/${blockId}/translation`);
    if (cached.ok) {
      const data = await cached.json();
      el.translationText.textContent = data.block.text || "";
    }
  } catch {}
}

async function translateSelected() {
  if (!state.selectedBlock) return;
  saveSettings();
  if (!el.baseUrl.value.trim() || !el.modelName.value.trim()) {
    el.translationText.textContent = "Base URL과 Model을 먼저 입력하세요.";
    return;
  }
  el.translateBtn.disabled = true;
  el.translateBtn.textContent = "번역 중";
  el.translationText.textContent = "요청 중...";
  try {
    const res = await fetch(`/api/jobs/${state.jobId}/blocks/${state.selectedBlock.id}/translate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_url: el.baseUrl.value.trim(),
        model: el.modelName.value.trim(),
        api_key: el.apiKey.value.trim() || null,
        temperature: 0,
        json_mode: el.jsonMode.checked,
        task: el.taskInput.value,
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    el.translationText.textContent = data.block.text || "";
  } catch (error) {
    el.translationText.textContent = `오류: ${error.message}`;
  } finally {
    el.translateBtn.disabled = false;
    el.translateBtn.textContent = "선택 영역 번역";
  }
}

function bindEvents() {
  el.dropZone.addEventListener("click", () => el.pdfInput.click());
  el.dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") el.pdfInput.click();
  });
  el.pdfInput.addEventListener("change", () => {
    const file = el.pdfInput.files?.[0];
    if (file) uploadPdf(file).catch((error) => {
      setStatus("error", "error");
      el.docMeta.textContent = error.message;
    });
  });
  for (const eventName of ["dragenter", "dragover"]) {
    el.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      el.dropZone.classList.add("dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    el.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      el.dropZone.classList.remove("dragging");
    });
  }
  el.dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files?.[0];
    if (file) uploadPdf(file).catch((error) => {
      setStatus("error", "error");
      el.docMeta.textContent = error.message;
    });
  });
  el.zoomOut.addEventListener("click", () => {
    state.zoom = Math.max(0.6, state.zoom - 0.1);
    renderPdf();
  });
  el.zoomIn.addEventListener("click", () => {
    state.zoom = Math.min(2.4, state.zoom + 0.1);
    renderPdf();
  });
  el.translateBtn.addEventListener("click", translateSelected);
  el.closeDetail.addEventListener("click", () => {
    el.detailContent.classList.add("hidden");
    el.detailEmpty.classList.remove("hidden");
    state.selectedBlock = null;
    document.querySelectorAll(".block-box.selected").forEach((node) => node.classList.remove("selected"));
  });
  [el.baseUrl, el.modelName, el.protectedTerms, el.taskInput, el.jsonMode].forEach((node) => {
    node.addEventListener("change", saveSettings);
  });
}

loadSettings();
bindEvents();
