const MODEL_ROOT_URLS = [
  // "/AFTER/export_onnx",        // GitHub Pages
  // "/after",
  // "../export_onnx",  
  "/export_onnx",             // Local development
  // "/web_onnx_app/export_onnx"  // Fallback
];
const MODEL_FILE = "midi_full_audio.onnx";
const MODEL_DATA_FILE = "midi_full_audio.onnx.data";
const MAP_IMAGE_FILE = "map.png";
const DEFAULT_MAP_IMAGE_URL = "../docs/background_transparent.png"
const MODEL_CACHE_PREFIX = "after-midi-onnx-v4";
const DEFAULT_MODEL_NAME = "orchestral_simdino";
const SAMPLE_RATE = 44100;
const CHUNK_SAMPLES = 262144;
const CHUNK_SECONDS = CHUNK_SAMPLES / SAMPLE_RATE;
const MAP_RANGE = 1.25;

const state = {
  session: null,
  midi: null,
  notes: [],
  mapImage: null,
  mapPoint01: { x: 0.5, y: 0.5 },
  modelBaseUrl: null,
  selectedModelName: "",
  loadedModelName: "",
  availableModels: [],
  mapObjectUrl: "",
  generations: [],
  generationIndex: 0,
  noteSource: "sequencer",
  importedModelData: new Map(),
  inputDimsByName: {},
  noiseDims: null,
  pianoRollDims: null,
  pianoRollInputName: null,
  _drawMapCallCount: 0,
  _defaultMapLoadAbort: null,
};

const el = {
  modelStatus: document.getElementById("modelStatus"),
  generateButton: document.getElementById("generateButton"),
  midiFile: document.getElementById("midiFile"),
  durationSelect: document.getElementById("durationSelect"),
  startTime: document.getElementById("startTime"),
  midiCanvas: document.getElementById("midiCanvas"),
  midiMeta: document.getElementById("midiMeta"),
  sequenceNote: document.getElementById("sequenceNote"),
  sequenceOctave: document.getElementById("sequenceOctave"),
  progressionSelect: document.getElementById("progressionSelect"),
  customProgression: document.getElementById("customProgression"),
  chordsPerChunk: document.getElementById("chordsPerChunk"),
  arpEnabled: document.getElementById("arpEnabled"),
  arpMode: document.getElementById("arpMode"),
  notesPerChord: document.getElementById("notesPerChord"),
  notesArpeggiated: document.getElementById("notesArpeggiated"),
  velocityMin: document.getElementById("velocityMin"),
  velocityMax: document.getElementById("velocityMax"),
  velocityMinVal: document.getElementById("velocityMinVal"),
  velocityMaxVal: document.getElementById("velocityMaxVal"),
  velocityVariance: document.getElementById("velocityVariance"),
  velocityVarianceVal: document.getElementById("velocityVarianceVal"),
  rerollVelocityButton: document.getElementById("rerollVelocityButton"),
  rerollArpButton: document.getElementById("rerollArpButton"),
  mapCanvas: document.getElementById("mapCanvas"),
  mapWrap: document.getElementById("mapWrap"),
  crosshair: document.getElementById("crosshair"),
  coords: document.getElementById("coords"),
  historyList: document.getElementById("historyList"),
  historyEmpty: document.getElementById("historyEmpty"),
  consoleLog: document.getElementById("consoleLog"),
  status: document.getElementById("status"),
  modelSelect: document.getElementById("modelSelect"),
  loadModelButton: document.getElementById("loadModelButton"),
};

ort.env.wasm.numThreads = 1;


function makeSessionOptions(executionProviders, dataBytes) {
  return {
    executionProviders,
    graphOptimizationLevel: "all",
    externalData: [
      {
        path: "midi_full_audio.onnx.data",
        data: dataBytes,
      },
    ],
  };
}

function getBackendPreference() {
  const value = new URLSearchParams(window.location.search).get("backend");
  if (value === "webgpu" || value === "wasm") {
    return value;
  }
  return "auto";
}

async function getWebGpuAdapter() {
  if (!window.isSecureContext) {
    appendConsole(`WebGPU unavailable: ${window.location.origin} is not a secure context. Use HTTPS or open the app through localhost.`);
    return null;
  }

  if (!navigator.gpu) {
    appendConsole("WebGPU unavailable: navigator.gpu is missing.");
    return null;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      appendConsole("WebGPU unavailable: requestAdapter() returned null.");
      return null;
    }
    appendConsole("WebGPU adapter available.");
    return adapter;
  } catch (error) {
    appendConsole(`WebGPU unavailable: ${error.message || String(error)}`);
    return null;
  }
}

async function createOnnxSession(modelFiles) {
  const backendPreference = getBackendPreference();

  if (backendPreference === "wasm") {
    appendConsole("ONNX execution provider: wasm");
    return await ort.InferenceSession.create(
      modelFiles.modelBytes,
      makeSessionOptions(["wasm"], modelFiles.dataBytes),
    );
  }

  const webGpuAdapter = await getWebGpuAdapter();
  if (webGpuAdapter) {
    appendConsole("ONNX execution provider: webgpu");
    try {
      return await ort.InferenceSession.create(
        modelFiles.modelBytes,
        makeSessionOptions(["webgpu"], modelFiles.dataBytes),
      );
    } catch (error) {
      appendConsole(`WebGPU session failed: ${error.message || String(error)}`);
      if (backendPreference === "webgpu") {
        throw error;
      }
    }
  } else if (backendPreference === "webgpu") {
    throw new Error("WebGPU was requested with ?backend=webgpu, but no adapter is available.");
  }

  appendConsole("ONNX execution provider: wasm");
  return await ort.InferenceSession.create(
    modelFiles.modelBytes,
    makeSessionOptions(["wasm"], modelFiles.dataBytes),
  );
}


function modelCacheName(modelName) {
  return `${MODEL_CACHE_PREFIX}-${modelName}`;
}

el.loadModelButton.addEventListener("click", () => {
  loadSelectedModel().catch((error) => {
    console.error(error);
    setStatus(error.message || String(error), true);
    setModelControlsBusy(false);
    refreshGenerateState();
  });
});
el.modelSelect.addEventListener("change", () => {
  state.selectedModelName = el.modelSelect.value;
  const model = getSelectedModel();
  state.modelBaseUrl = model?.baseUrl || null;
  el.loadModelButton.disabled = !model;
  if (state.loadedModelName !== state.selectedModelName) {
    setStatus("Load the selected model.");
  }
  refreshGenerateState();
});

const dropZone = document.getElementById("dropZone");

dropZone?.addEventListener("dragover", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.add("active");
});

dropZone?.addEventListener("dragleave", (e) => {
  e.preventDefault();
  dropZone.classList.remove("active");
});

dropZone?.addEventListener("drop", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.remove("active");
  handleDroppedFiles(e.dataTransfer.items || e.dataTransfer.files);
});

async function readDirectoryRecursive(dirEntry, requiredFiles, fileMap) {
  const reader = dirEntry.createReader();

  return new Promise((resolve, reject) => {
    const readEntries = () => {
      reader.readEntries(
        async (entries) => {
          for (const entry of entries) {
            if (entry.isDirectory) {
              await readDirectoryRecursive(entry, requiredFiles, fileMap);
            } else if (entry.isFile) {
              const file = await new Promise((resolve, reject) => {
                entry.file(resolve, reject);
              });
              if (requiredFiles.includes(file.name)) {
                fileMap.set(file.name, file);
                appendConsole(`Found: ${file.name}`);
              }
            }
          }
          if (entries.length < 100) {
            resolve();
          } else {
            readEntries();
          }
        },
        reject
      );
    };
    readEntries();
  });
}

async function handleDroppedFiles(items) {
  appendConsole("Files/folders dropped, scanning for model...");
  setModelControlsBusy(true);
  setStatus("Importing dropped files...");

  const requiredFiles = [MODEL_FILE, MODEL_DATA_FILE, MAP_IMAGE_FILE];
  const fileMap = new Map();
  let folderName = "";

  try {
    if (items instanceof DataTransferItemList) {
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind === "file") {
          const entry = item.webkitGetAsEntry?.();
          if (entry) {
            if (entry.isDirectory) {
              appendConsole(`Found directory: ${entry.name}`);
              folderName = entry.name;
              await readDirectoryRecursive(entry, requiredFiles, fileMap);
            } else {
              const file = item.getAsFile();
              if (file && requiredFiles.includes(file.name)) {
                fileMap.set(file.name, file);
                appendConsole(`Found: ${file.name}`);
              }
            }
          } else {
            const file = item.getAsFile();
            if (file && requiredFiles.includes(file.name)) {
              fileMap.set(file.name, file);
              appendConsole(`Found: ${file.name}`);
            }
          }
        }
      }
    } else {
      for (let i = 0; i < items.length; i++) {
        const file = items[i];
        if (requiredFiles.includes(file.name)) {
          fileMap.set(file.name, file);
          appendConsole(`Found: ${file.name}`);
        }
      }
    }

    if (fileMap.size === 0) {
      setModelControlsBusy(false);
      setStatus("No model files found in dropped items");
      return;
    }

    folderName = `imported_${Date.now()}`;

    const missing = requiredFiles.filter(f => !fileMap.has(f));
    if (missing.length > 0) {
      setModelControlsBusy(false);
      setStatus(`Missing files: ${missing.join(", ")}`);
      return;
    }

    await performImport(folderName, fileMap, requiredFiles);
  } catch (error) {
    appendConsole(`Drop import error: ${error.message}`);
    setStatus(error.message || String(error), true);
    setModelControlsBusy(false);
  }
}


function setStatus(message, error = false) {
  el.status.textContent = message;
  el.status.classList.toggle("is-error", error);
}

function appendConsole(message) {
  const line = document.createElement("div");
  line.className = "console-line";
  line.textContent = message;
  el.consoleLog.append(line);
  while (el.consoleLog.children.length > 80) {
    el.consoleLog.firstChild.remove();
  }
  el.consoleLog.scrollTop = el.consoleLog.scrollHeight;
}

function refreshGenerateState() {
  el.generateButton.disabled = !(state.session && state.loadedModelName === state.selectedModelName && getActiveNotes().length);
}

function setGenerateBusy(isBusy) {
  el.generateButton.classList.toggle("is-generating", isBusy);
  el.generateButton.textContent = isBusy ? "Generating..." : "Generate";
  el.generateButton.setAttribute("aria-busy", isBusy ? "true" : "false");
}

function waitForPaint() {
  return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

function modelDisplayName(name = state.loadedModelName || state.selectedModelName) {
  return name || "after-midi";
}

function getSelectedModel() {
  return state.availableModels.find((model) => model.name === state.selectedModelName) || null;
}

function setModelControlsBusy(isBusy) {
  el.modelSelect.disabled = isBusy || state.availableModels.length === 0;
  el.loadModelButton.disabled = isBusy || !state.selectedModelName;
}

function releaseCurrentModel() {
  appendConsole("releaseCurrentModel called - clearing map");
  const oldSession = state.session;
  state.session = null;
  state.loadedModelName = "";
  state.mapImage = null;
  if (state.mapObjectUrl) {
    appendConsole(`Revoking mapObjectUrl: ${state.mapObjectUrl}`);
    URL.revokeObjectURL(state.mapObjectUrl);
    state.mapObjectUrl = "";
  }

  try {
    oldSession?.release?.();
  } catch (e) {
    console.warn("Could not release old ONNX session", e);
  }
}
async function hasModelFiles(baseUrl) {
  const required = [MODEL_FILE, MODEL_DATA_FILE, MAP_IMAGE_FILE];

  for (const file of required) {
    try {
      const response = await fetch(`${baseUrl}/${file}`, {
        method: "GET",
        cache: "no-store",
        headers: { Range: "bytes=0-0" },
      });

      if (!response.ok && response.status !== 206) {
        appendConsole(`Missing ${baseUrl}/${file}: HTTP ${response.status}`);
        return false;
      }
    } catch (error) {
      appendConsole(`Missing ${baseUrl}/${file}: ${error.message || String(error)}`);
      return false;
    }
  }

  return true;
}

async function logDirectoryFiles(baseUrl) {
  appendConsole("Trying to list the files");
  try {
    const response = await fetch(`${baseUrl}/`, { cache: "no-store" });
    if (!response.ok) {
      appendConsole(`Could not list ${baseUrl}/: HTTP ${response.status}`);
      return;
    }

    const html = await response.text();
    const doc = new DOMParser().parseFromString(html, "text/html");
    const folderPath = new URL(`${baseUrl}/`, location.origin).pathname.replace(/\/$/, "");
    const entries = [...doc.querySelectorAll("a[href]")]
      .map((link) => link.getAttribute("href"))
      .filter(Boolean)
      .map((href) => new URL(href, response.url).pathname)
      .filter((pathname) => pathname.startsWith(`${folderPath}/`))
      .map((pathname) => pathname.slice(folderPath.length + 1).replace(/\/$/, ""))
      .filter((name) => name && !name.includes("/"));

    appendConsole(`Files in ${baseUrl}: ${entries.length ? entries.join(", ") : "(none)"}`);
  } catch (error) {
    appendConsole(`Could not list ${baseUrl}/: ${error.message || String(error)}`);
  }
}

async function scanModels() {
  console.log("SCAN MODELS CALLED");
  appendConsole("Scanning models...");

  const byName = new Map();
  const failures = [];

  for (const rootUrl of MODEL_ROOT_URLS) {
    try {
      appendConsole(`Scanning ${rootUrl}`);

      const response = await fetch(`${rootUrl}/`, { cache: "no-store" });
      if (!response.ok) {
        failures.push(`${rootUrl}: HTTP ${response.status}`);
        continue;
      }

      const html = await response.text();
      console.log(`RAW HTML from ${rootUrl}:`);
      appendConsole(html.slice(0, 500)); // first 500 chars only
      const doc = new DOMParser().parseFromString(html, "text/html");

      const links = [...doc.querySelectorAll("a[href]")];
      appendConsole(`${rootUrl}: ${links.length} links found`);

      for (const link of links) {
        const href = link.getAttribute("href");

        if (!href || href.startsWith("?") || href.startsWith("#")) {
          continue;
        }

        const url = new URL(href, response.url);
        const pathname = url.pathname.replace(/\/$/, "");
        const rootPath = new URL(rootUrl, location.href).pathname.replace(/\/$/, "");

        if (!pathname.startsWith(rootPath)) {
          continue;
        }

        const parts = pathname.split("/").filter(Boolean);
        const name = decodeURIComponent(parts[parts.length - 1] || "");

        if (!name || name === "export_onnx") {
          continue;
        }

        const baseUrl = pathname;

        appendConsole(`Checking model candidate: ${name} at ${baseUrl}`);
        await logDirectoryFiles(baseUrl);

        if (!byName.has(name) && await hasModelFiles(baseUrl)) {
          byName.set(name, { name, baseUrl });
          appendConsole(`Found model: ${name}`);
        }
      }
    } catch (error) {
      failures.push(`${rootUrl}: ${error.message || String(error)}`);
      appendConsole(`${rootUrl}: ${error.message || String(error)}`);
    }
  }

  state.availableModels = Array.from(byName.values())
    .sort((a, b) => a.name.localeCompare(b.name));

  renderModelOptions();

  if (!state.availableModels.length) {
    // el.modelStatus.textContent = "No models found";
    setStatus(`No valid models found. ${failures.join("; ")}`, true);
    appendConsole("No valid models found.");
    return;
  }

  const defaultModel =
    state.availableModels.find((model) => model.name === DEFAULT_MODEL_NAME) ||
    state.availableModels[0];

  state.selectedModelName = defaultModel.name;
  state.modelBaseUrl = defaultModel.baseUrl;
  el.modelSelect.value = defaultModel.name;

  setModelControlsBusy(false);
  refreshGenerateState();

  // el.modelStatus.textContent = `Selected model: ${defaultModel.name}`;
  setStatus("Load the selected model.");
  appendConsole(`Available models: ${state.availableModels.map((m) => m.name).join(", ")}`);
}
function renderModelOptions() {
  el.modelSelect.replaceChildren();
  if (!state.availableModels.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No models found";
    el.modelSelect.append(option);
    setModelControlsBusy(false);
    return;
  }

  for (const model of state.availableModels) {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.name;
    el.modelSelect.append(option);
  }
}

async function importLocalModel() {
  appendConsole("=== IMPORT START ===");
  setModelControlsBusy(true);
  setStatus("Importing model from local folder...");

  const requiredFiles = [MODEL_FILE, MODEL_DATA_FILE, MAP_IMAGE_FILE];
  let fileMap = new Map();
  let folderName = "";

  appendConsole(`showDirectoryPicker available: ${"showDirectoryPicker" in window}`);

  if ("showDirectoryPicker" in window) {
    appendConsole("Using showDirectoryPicker API");
    let dirHandle;
    try {
      dirHandle = await window.showDirectoryPicker();
    } catch (error) {
      if (error.name === "AbortError") {
        setModelControlsBusy(false);
        setStatus("Import cancelled.");
        return;
      }
      throw new Error(`Failed to open folder: ${error.message}`);
    }

    folderName = dirHandle.name;
    appendConsole(`Scanning folder: ${folderName}`);

    try {
      for await (const [name, handle] of dirHandle.entries()) {
        if (handle.kind === "file" && requiredFiles.includes(name)) {
          fileMap.set(name, await handle.getFile());
          appendConsole(`Found: ${name}`);
        }
      }
    } catch (error) {
      setModelControlsBusy(false);
      throw new Error(`Failed to read folder: ${error.message}`);
    }

    const missing = requiredFiles.filter(f => !fileMap.has(f));
    if (missing.length > 0) {
      setModelControlsBusy(false);
      throw new Error(`Missing required files: ${missing.join(", ")}`);
    }

    await performImport(folderName, fileMap, requiredFiles);
    return;
  } else {
    appendConsole("Using webkitdirectory fallback");
    let input = document.getElementById("__importModelFileInput");
    if (!input) {
      appendConsole("Creating file input element");
      input = document.createElement("input");
      input.id = "__importModelFileInput";
      input.type = "file";
      input.webkitdirectory = true;
      input.multiple = true;
      input.style.display = "none";
      document.body.appendChild(input);
    }

    return new Promise((resolve, reject) => {
      appendConsole("Waiting for file selection");
      input.onchange = async () => {
        try {
          const files = Array.from(input.files || []);
          if (!files.length) {
            setModelControlsBusy(false);
            setStatus("Import cancelled.");
            resolve();
            return;
          }

          const filesByName = new Map(files.map(f => [f.name, f]));
          folderName = files[0].webkitRelativePath?.split("/")[0] || "imported_model";
          appendConsole(`Scanning folder: ${folderName}`);

          for (const fileName of requiredFiles) {
            const file = filesByName.get(fileName);
            if (file) {
              fileMap.set(fileName, file);
              appendConsole(`Found: ${fileName}`);
            }
          }

          if (fileMap.size === requiredFiles.length) {
            await performImport(folderName, fileMap, requiredFiles);
            resolve();
          } else {
            const missing = requiredFiles.filter(f => !fileMap.has(f));
            reject(new Error(`Missing required files: ${missing.join(", ")}`));
          }
        } catch (error) {
          reject(error);
        } finally {
          input.value = "";
        }
      };

      input.click();
    });
  }
}

async function performImport(folderName, fileMap, requiredFiles) {
  setStatus("Processing model files...");

  const baseUrl = `/api/custom-models/${folderName}`;
  const modelData = {};

  const useCache = "caches" in window;
  if (useCache) {
    setStatus("Caching model files...");
    const cacheName = modelCacheName(folderName);
    const cache = await caches.open(cacheName);

    for (const fileName of requiredFiles) {
      const file = fileMap.get(fileName);
      try {
        const arrayBuffer = await file.arrayBuffer();
        const response = new Response(arrayBuffer, {
          headers: { "Content-Type": file.type || "application/octet-stream" },
        });
        const cacheKey = new Request(`${baseUrl}/${fileName}`);
        await cache.put(cacheKey, response);
        appendConsole(`Cached: ${baseUrl}/${fileName}`);
      } catch (error) {
        throw new Error(`Failed to cache ${fileName}: ${error.message}`);
      }
    }
  } else {
    appendConsole("Cache API unavailable. Storing model in memory (session only).");
    for (const fileName of requiredFiles) {
      const file = fileMap.get(fileName);
      try {
        const arrayBuffer = await file.arrayBuffer();
        modelData[fileName] = arrayBuffer;
        appendConsole(`Loaded to memory: ${fileName}`);
      } catch (error) {
        throw new Error(`Failed to read ${fileName}: ${error.message}`);
      }
    }
    state.importedModelData.set(folderName, modelData);
    appendConsole(`Stored imported model data: ${Object.keys(modelData).join(", ")}`);
  }

  const model = { name: folderName, baseUrl };

  const existingIndex = state.availableModels.findIndex(m => m.name === folderName);
  if (existingIndex >= 0) {
    state.availableModels[existingIndex] = model;
    appendConsole(`Updated existing model: ${folderName}`);
  } else {
    state.availableModels.push(model);
    state.availableModels.sort((a, b) => a.name.localeCompare(b.name));
    appendConsole(`Added new model: ${folderName}`);
  }

  renderModelOptions();
  state.selectedModelName = folderName;
  state.modelBaseUrl = baseUrl;
  el.modelSelect.value = folderName;

  setStatus(`Model imported: ${folderName}`);
  setModelControlsBusy(false);
  refreshGenerateState();
}

async function isModelFullyCached(model) {
  if (!("caches" in window)) {
    return false;
  }

  const cache = await caches.open(modelCacheName(model.name));
  const urls = [MODEL_FILE, MODEL_DATA_FILE, MAP_IMAGE_FILE].map((file) => `${model.baseUrl}/${file}`);
  const matches = await Promise.all(urls.map((url) => cache.match(new Request(url))));
  return matches.every(Boolean);
}

async function loadSelectedModel(forceReload = null) {
  const model = getSelectedModel();
  if (!model) {
    throw new Error("Select a model first.");
  }
  setModelControlsBusy(true);
  const shouldRefreshCache = forceReload ?? !(await isModelFullyCached(model));
  if (shouldRefreshCache) {
    appendConsole(`Refreshing cached model: ${model.name}`);
    setStatus("Refreshing cached model...");
  } else {
    setStatus("Loading selected model...");
  }
  releaseCurrentModel();
  state.selectedModelName = model.name;
  state.modelBaseUrl = model.baseUrl;
  await loadModel(model, shouldRefreshCache);
  setModelControlsBusy(false);
}

async function loadModel(model, forceReload = false) {
  try {
    await registerServiceWorker();
    const modelFiles = await warmModelCache(model, forceReload);
    await loadMapImage(model, forceReload);
    appendConsole("Creating ONNX session...");
    state.session = await createOnnxSession(modelFiles);

    console.log("inputNames:", state.session.inputNames);
    console.log("inputMetadata:", state.session.inputMetadata);

    state.inputDimsByName = {};
    state.noiseDims = null;
    state.pianoRollDims = null;
    state.pianoRollInputName = null;

    // for (const inputName of state.session.inputNames) {
    //   try {
    //     const dims = getInputDims(inputName, null);
    //     appendConsole(`${inputName} dims: [${dims.join(", ")}]`);
    //   } catch (error) {
    //     appendConsole(`${inputName} dims unavailable: ${error.message}`);
    //   }
    // }


    state.loadedModelName = model.name;
    // `Model ready: ${model.name}`;
    appendConsole(`Model ready (${model.name}): ${state.session.inputNames.join(", ")} -> ${state.session.outputNames.join(", ")}`);
    setStatus("Model ready. Choose notes and a map position.");
  } catch (error) {
    console.error(error);
    // el.modelStatus.textContent = "Model failed to load";
    setStatus(error.message || String(error), true);
    throw error;
  } finally {
    refreshGenerateState();
  }
}

async function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    return;
  }
  try {
    const basePath = window.location.pathname.includes('/AFTER/') ? '/AFTER/' : '/';
    await navigator.serviceWorker.register(`${basePath}sw.js`, { scope: basePath });
    await navigator.serviceWorker.ready;
  } catch (error) {
    console.warn("Service worker registration failed", error);
  }
}
async function warmModelCache(model, forceReload = false) {
  const modelUrl = `${model.baseUrl}/${MODEL_FILE}`;
  const dataUrl = `${model.baseUrl}/${MODEL_DATA_FILE}`;
  const mapUrl = `${model.baseUrl}/${MAP_IMAGE_FILE}`;

  const importedData = state.importedModelData.get(model.name);
  if (importedData) {
    appendConsole("Using imported model from memory");
    return {
      modelBytes: new Uint8Array(importedData[MODEL_FILE]),
      dataBytes: new Uint8Array(importedData[MODEL_DATA_FILE]),
    };
  }

  if (!("caches" in window)) {
    appendConsole("Cache API unavailable, downloading model...");
    const modelBytes = await fetchBytes(modelUrl, forceReload);
    const dataBytes = await fetchBytes(dataUrl, forceReload);
    return { modelBytes, dataBytes };
  }

  const cacheName = modelCacheName(model.name);
  const cache = await caches.open(cacheName);

  appendConsole(`Using cache: ${cacheName}`);

  const modelBytes = await cacheWithProgress(cache, modelUrl, "model graph", forceReload);
  const dataBytes = await cacheWithProgress(cache, dataUrl, "model weights", forceReload);
  await cacheWithProgress(cache, mapUrl, "map image", forceReload);

  return { modelBytes, dataBytes };
}
async function cacheWithProgress(cache, url, label, forceReload = false) {
  const cacheKey = new Request(url);

  if (!forceReload) {
    const cached = await cache.match(cacheKey);
    if (cached) {
      appendConsole(`${label} loaded from cache`);
      return new Uint8Array(await cached.arrayBuffer());
    }
  }

  appendConsole(`Downloading ${label}...`);

  const response = await fetch(url, {
    cache: forceReload ? "reload" : "default",
  });

  if (!response.ok) {
    throw new Error(`Could not download ${url}: HTTP ${response.status}`);
  }

  const bytes = new Uint8Array(await response.arrayBuffer());

  await cache.put(cacheKey, new Response(bytes.slice(0), {
    status: 200,
    headers: {
      "Content-Type": response.headers.get("Content-Type") || "application/octet-stream",
    },
  }));

  appendConsole(`${label} cached`);

  return bytes;
}

async function fetchBytes(url, forceReload = false) {
  const response = await fetch(url, { cache: forceReload ? "reload" : "force-cache" });
  if (!response.ok) {
    throw new Error(`Could not download ${url}: HTTP ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}
async function loadMapImage(model, forceReload = false) {
  const mapUrl = `${model.baseUrl}/${MAP_IMAGE_FILE}`;

  let blob;

  try {
    const importedData = state.importedModelData.get(model.name);
    appendConsole(`Looking for imported map: model.name=${model.name}, importedData=${!!importedData}, has map=${!!importedData?.[MAP_IMAGE_FILE]}`);
    if (importedData && importedData[MAP_IMAGE_FILE]) {
      appendConsole("map image loaded from imported model");
      const mapArrayBuffer = importedData[MAP_IMAGE_FILE];
      appendConsole(`map arrayBuffer size: ${mapArrayBuffer.byteLength} bytes`);
      blob = new Blob([mapArrayBuffer], { type: "image/png" });
      appendConsole(`created blob size: ${blob.size} bytes`);
    } else if ("caches" in window) {
      const cache = await caches.open(modelCacheName(model.name));
      const cached = !forceReload ? await cache.match(mapUrl) : null;

      if (cached) {
        appendConsole("map image loaded from cache");
        blob = await cached.blob();
      } else {
        const response = await fetch(mapUrl, {
          cache: forceReload ? "reload" : "default",
        });

        if (!response.ok) {
          throw new Error(`map image HTTP ${response.status}`);
        }

        blob = await response.blob();

        await cache.put(mapUrl, new Response(blob.slice(0), {
          status: 200,
          headers: {
            "Content-Type": response.headers.get("Content-Type") || "image/png",
          },
        }));
      }
    } else {
      const response = await fetch(mapUrl, {
        cache: forceReload ? "reload" : "force-cache",
      });

      if (!response.ok) {
        throw new Error(`map image HTTP ${response.status}`);
      }

      blob = await response.blob();
    }
  } catch (error) {
    appendConsole(`map image unavailable, using default image: ${error.message || String(error)}`);

    const fallbackResponse = await fetch(DEFAULT_MAP_IMAGE_URL, {
      cache: "force-cache",
    });

    if (!fallbackResponse.ok) {
      throw new Error(`Could not load default map image: HTTP ${fallbackResponse.status}`);
    }

    blob = await fallbackResponse.blob();
  }

  const img = new Image();

  if (state.mapObjectUrl) {
    URL.revokeObjectURL(state.mapObjectUrl);
  }

  state.mapObjectUrl = URL.createObjectURL(blob);
  appendConsole(`Created map object URL: ${state.mapObjectUrl}`);
  img.src = state.mapObjectUrl;

  try {
    await img.decode();
    appendConsole("Map image decoded successfully");
  } catch (decodeError) {
    appendConsole(`Warning: map image decode failed: ${decodeError.message}, but continuing`);
  }

  state.mapImage = img;
  appendConsole(`Set state.mapImage, calling drawMap()`);
  drawMap();
}

async function loadMidiFile(file) {
  const arrayBuffer = await file.arrayBuffer();
  const midi = new Midi(arrayBuffer);
  const notes = [];
  for (const track of midi.tracks) {
    for (const note of track.notes) {
      notes.push({
        midi: note.midi,
        time: note.time,
        duration: note.duration,
        velocity: note.velocity ?? 1,
      });
    }
  }
  notes.sort((a, b) => a.time - b.time);
  state.midi = midi;
  state.notes = notes;
  const duration = Math.max(midi.duration || 0, ...notes.map((n) => n.time + n.duration), 0);
  el.midiMeta.textContent = `${file.name} | ${notes.length} notes | ${duration.toFixed(2)} s`;
  updateSourceUi();
  drawMidiPreview();
  setStatus("MIDI loaded. Choose a map position and generate.");
  refreshGenerateState();
}

function cloneNotes(notes) {
  return notes.map((note) => ({ ...note }));
}

function snapshotGenerationSettings() {
  return {
    modelName: state.loadedModelName,
    source: state.noteSource,
    durationChunks: el.durationSelect.value,
    startTime: el.startTime.value,
    mapPoint01: { ...state.mapPoint01 },
    midi: {
      notes: cloneNotes(state.notes),
      meta: el.midiMeta.textContent,
    },
    sequencer: {
      sequenceNote: el.sequenceNote.value,
      sequenceOctave: el.sequenceOctave.value,
      progressionSelect: el.progressionSelect.value,
      customProgression: el.customProgression.value,
      chordsPerChunk: el.chordsPerChunk.value,
      arpEnabled: el.arpEnabled.checked,
      arpMode: el.arpMode.value,
      notesPerChord: el.notesPerChord.value,
      notesArpeggiated: el.notesArpeggiated.value,
      velocityMin: el.velocityMin.value,
      velocityMax: el.velocityMax.value,
      velocityVariance: el.velocityVariance.value,
    },
  };
}

function getActiveNotes() {
  if (state.noteSource === "sequencer") {
    const chunks = Number(el.durationSelect.value) || 1;
    return buildSequencerNotes(chunks * CHUNK_SECONDS);
  }
  return state.notes;
}

function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function numberOrDefault(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function repeatToLength(values, length) {
  if (!values.length || length <= 0) {
    return [];
  }
  return Array.from({ length }, (_, index) => values[index % values.length]);
}

function buildSequencerNotes(totalEndSeconds, forPreview = false) {
  const noteNum = Number(el.sequenceNote.value) || 0;
  const octave = Number(el.sequenceOctave.value) || 3;
  const base = (octave + 1) * 12 + noteNum;

  const chunks = Number(el.durationSelect.value) || 1;
  const chordsPerChunk = Math.max(1, Math.floor(Number(el.chordsPerChunk.value) || 1));
  const chordDuration = CHUNK_SECONDS / chordsPerChunk;
  const totalChords = chunks * chordsPerChunk;
  const progression = getProgressionTokens();

  const arpMode = el.arpEnabled.checked ? el.arpMode.value : "off";
  const notesPerChordVal = el.notesPerChord.value;
  const velMin = numberOrDefault(el.velocityMin.value, 64);
  const velMax = numberOrDefault(el.velocityMax.value, 115);
  const velMean = (velMin + velMax) / 2;
  const variance = numberOrDefault(el.velocityVariance.value, 0.5);

  const getVel = () => {
    if (forPreview) return clamp01(velMean / 127);
    const uniformRandom = velMin + Math.random() * (velMax - velMin);
    const vel127 = velMean * (1 - variance) + uniformRandom * variance;
    return clamp01(vel127 / 127);
  };

  const notes = [];
  for (let i = 0; i < totalChords; i++) {
    const chordStart = i * chordDuration;
    if (chordStart >= totalEndSeconds) break;

    const fullChord = chordForToken(base, progression[i % progression.length]);
    const n = notesPerChordVal === "random"
      ? Math.floor(Math.random() * 6) + 1
      : Math.min(Number(notesPerChordVal) || 3, fullChord.length);

    let chord = fullChord.slice(0, n);

    if (arpMode === "down") {
      chord = [...chord].reverse();
    } else if (arpMode === "random") {
      chord = shuffleArray([...chord]);
    } else if (arpMode === "alt") {
      const up = [...chord];
      const down = [...chord].reverse().slice(1, -1);
      chord = [...up, ...down];
    }

    if (arpMode !== "off") {
      const m = Number(el.notesArpeggiated.value) || 2;
      chord = repeatToLength(chord, m);
      const step = chordDuration / Math.max(1, chord.length);
      chord.forEach((midi, idx) => {
        notes.push({ midi, time: chordStart + idx * step, duration: step * 0.92, velocity: getVel() });
      });
    } else {
      for (const midi of chord) {
        notes.push({ midi, time: chordStart, duration: chordDuration * 0.94, velocity: getVel() });
      }
    }
  }
  return notes;
}

function getProgressionTokens() {
  const value = el.progressionSelect.value === "custom"
    ? el.customProgression.value
    : el.progressionSelect.value;
  const tokens = value
    .replaceAll("-", " ")
    .split(/\s+/)
    .map((token) => token.trim())
    .filter(Boolean);
  return tokens.length ? tokens : ["I"];
}

function chordForToken(base, token) {
  const normalized = token.replace("°", "dim");
  const lower = normalized.toLowerCase();
  const roman = lower.replace(/[^iv]+/g, "");
  const degreeMap = { i: 0, ii: 2, iii: 4, iv: 5, v: 7, vi: 9, vii: 11 };
  const rootOffset = degreeMap[roman] ?? 0;
  const isDiminished = lower.includes("dim") || lower.includes("vii");
  const isMinor = token === token.toLowerCase() && !isDiminished;

  // Extended chord intervals up to 6 notes: root, 3rd, 5th, 7th, 9th, 11th
  let intervals;
  if (isDiminished) {
    intervals = [0, 3, 6, 9, 12, 15];
  } else if (isMinor) {
    intervals = [0, 3, 7, 10, 14, 17];
  } else {
    intervals = [0, 4, 7, 11, 14, 17];
  }

  const root = base + rootOffset + 12;
  return intervals.map((interval) => Math.max(0, Math.min(127, root + interval)));
}

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const ROLL_LEFT = 36;

function drawMidiPreview() {
  const canvas = el.midiCanvas;
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f5f6f8";
  ctx.fillRect(0, 0, width, height);

  const start = state.noteSource === "sequencer" ? 0 : Number(el.startTime.value) || 0;
  const chunks = Number(el.durationSelect.value) || 1;
  const duration = chunks * CHUNK_SECONDS;
  const end = start + duration;

  let activeNotes;
  if (state.noteSource === "sequencer") {
    activeNotes = buildSequencerNotes(chunks * CHUNK_SECONDS, false);
  } else {
    activeNotes = state.notes;
  }
  const visible = activeNotes.filter((n) => n.time < end && n.time + n.duration > start);

  // Store for hover detection
  state.midiPreviewData = { start, duration, visible, minPitch: 48, maxPitch: 84 };

  const plotW = width - ROLL_LEFT;

  let minPitch = 48;
  let maxPitch = 84;
  if (visible.length) {
    minPitch = Math.max(0, Math.min(...visible.map((n) => n.midi)) - 3);
    maxPitch = Math.min(127, Math.max(...visible.map((n) => n.midi)) + 3);
  }

  state.midiPreviewData.minPitch = minPitch;
  state.midiPreviewData.maxPitch = maxPitch;

  const N = maxPitch - minPitch + 1;
  const rowH = height / N;

  // Piano-key tint per row
  for (let p = minPitch; p <= maxPitch; p++) {
    const i = p - minPitch;
    const yTop = height - (i + 1) * rowH;
    const isBlack = [1, 3, 6, 8, 10].includes(p % 12);
    ctx.fillStyle = isBlack ? "rgba(0,0,0,0.06)" : "rgba(0,0,0,0.02)";
    ctx.fillRect(ROLL_LEFT, yTop, plotW, rowH);
  }

  // Horizontal note grid lines + C-note labels
  ctx.font = "10px ui-monospace, Menlo, Consolas, monospace";
  for (let p = minPitch; p <= maxPitch; p++) {
    const i = p - minPitch;
    const yTop = height - (i + 1) * rowH;
    if (p % 12 === 0) {
      ctx.strokeStyle = "rgba(0,0,0,0.15)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ROLL_LEFT, yTop);
      ctx.lineTo(width, yTop);
      ctx.stroke();
      const octNum = Math.floor(p / 12) - 1;
      ctx.fillStyle = "rgba(0,0,0,0.75)";
      ctx.fillText(`C${octNum}`, 2, height - (i + 0.5) * rowH + 3);
    } else if (N <= 24) {
      // show all note names when zoomed in
      ctx.strokeStyle = "rgba(0,0,0,0.06)";
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(ROLL_LEFT, yTop);
      ctx.lineTo(width, yTop);
      ctx.stroke();
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillText(NOTE_NAMES[p % 12], 2, height - (i + 0.5) * rowH + 3);
    }
  }

  // Vertical time grid (one line per second)
  ctx.strokeStyle = "rgba(0,0,0,0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= Math.ceil(duration); i++) {
    const x = ROLL_LEFT + (i / duration) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  // Left margin separator
  ctx.strokeStyle = "rgba(0,0,0,0.2)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(ROLL_LEFT, 0);
  ctx.lineTo(ROLL_LEFT, height);
  ctx.stroke();

  // Draw notes — velocity controls both color and opacity
  for (const note of visible) {
    const x0 = ROLL_LEFT + Math.max(0, (note.time - start) / duration) * plotW;
    const x1 = ROLL_LEFT + Math.min(1, (note.time + note.duration - start) / duration) * plotW;
    const i = note.midi - minPitch;
    const yTop = height - (i + 1) * rowH;

    const v = Math.max(0, Math.min(1, note.velocity ?? 0.7));

    const r = Math.round(30 - v * 15);   // 30 → 15
    const g = Math.round(45 - v * 20);   // 45 → 25
    const b = Math.round(90 - v * 20);   // 90 → 70

    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.35 + 0.5 * v})`;
    ctx.fillRect(x0, yTop + 0.5, Math.max(2, x1 - x0), Math.max(2, rowH - 1));
  }

  // Time labels
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  ctx.font = "11px ui-monospace, Menlo, Consolas, monospace";
  ctx.fillText(`${start.toFixed(2)}s`, ROLL_LEFT + 4, height - 4);
  ctx.fillText(`${end.toFixed(2)}s`, width - 56, height - 4);
}

function buildPianoRoll(windowStartSeconds) {
  const dims = getPianoRollDims();
  const channels = dims[1];
  const frames = dims[2];
  if (channels !== 128) {
    throw new Error(`Expected piano_roll pitch dimension 128, got ${channels}.`);
  }

  const roll = new Float32Array(channels * frames);
  const frameSeconds = CHUNK_SECONDS / frames;
  const windowEnd = windowStartSeconds + CHUNK_SECONDS;
  for (const note of getActiveNotes()) {
    const noteStart = note.time;
    const noteEnd = note.time + note.duration;
    if (noteStart >= windowEnd || noteEnd <= windowStartSeconds) {
      continue;
    }
    const first = Math.max(0, Math.floor((noteStart - windowStartSeconds) / frameSeconds));
    const last = Math.min(frames - 1, Math.ceil((noteEnd - windowStartSeconds) / frameSeconds));
    const value = clamp01(note.velocity ?? 1);
    for (let frame = first; frame <= last; frame++) {
      roll[note.midi * frames + frame] = Math.max(roll[note.midi * frames + frame], value);
    }
  }
  return new ort.Tensor("float32", roll, dims);
}

function makeMapTensor() {
  const [x, y] = getMapPosition();
  return new ort.Tensor("float32", new Float32Array([x, y]), [1, 2]);
}

function getInputDims(name, fallbackDims = null) {
  if (state.inputDimsByName?.[name]) {
    return state.inputDimsByName[name];
  }

  const meta = getInputMeta(name);

  const dims =
    normalizeDims(meta?.dimensions) ||
    normalizeDims(meta?.dims) ||
    normalizeDims(meta?.shape);

  if (dims) {
    state.inputDimsByName[name] = dims;
    return dims;
  }

  if (fallbackDims) {
    appendConsole(`${name} metadata has no dims, using fallback [${fallbackDims.join(", ")}]`);
    state.inputDimsByName[name] = fallbackDims;
    return fallbackDims;
  }

  throw new Error(`Could not infer input dims for "${name}".`);
}
function makeNoiseTensor() {
  const dims = getNoiseDims();
  const size = dims.reduce((acc, value) => acc * value, 1);
  const data = new Float32Array(size);

  for (let i = 0; i < data.length; i += 2) {
    const u1 = Math.max(Math.random(), 1e-7);
    const u2 = Math.random();
    const r = Math.sqrt(-2 * Math.log(u1));

    data[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < data.length) {
      data[i + 1] = r * Math.sin(2 * Math.PI * u2);
    }
  }

  return new ort.Tensor("float32", data, dims);
}

function getNoiseDims() {
  if (state.noiseDims) {
    return state.noiseDims;
  }

  state.noiseDims = getInputDims("noise", null);
  // appendConsole(`noise dims from ONNX metadata: [${state.noiseDims.join(", ")}]`);

  return state.noiseDims;
}

function getPianoRollDims() {
  if (state.pianoRollDims) {
    return state.pianoRollDims;
  }

  const inputName = getPianoRollInputName();
  state.pianoRollDims = getInputDims(inputName, null);
  const noiseFrames = getNoiseDims()[2];
  const pianoRollFrames = state.pianoRollDims[2];
  appendConsole(`${inputName} frames from ONNX metadata: ${pianoRollFrames} (${pianoRollFrames / noiseFrames}x noise frames)`);

  return state.pianoRollDims;
}

function getPianoRollInputName() {
  if (state.pianoRollInputName) {
    return state.pianoRollInputName;
  }

  if (state.session?.inputNames.includes("piano_roll")) {
    state.pianoRollInputName = "piano_roll";
    return state.pianoRollInputName;
  }

  if (state.session?.inputNames.includes("time_cond")) {
    state.pianoRollInputName = "time_cond";
    appendConsole("Using legacy time_cond input as piano_roll.");
    return state.pianoRollInputName;
  }

  throw new Error("Model has no piano_roll input.");
}

function getInputMeta(name) {
  const metadata = state.session?.inputMetadata;

  if (!metadata) {
    return null;
  }

  // Case 1: Map
  if (metadata instanceof Map) {
    return metadata.get(name) || null;
  }

  // Case 2: Array
  if (Array.isArray(metadata)) {
    return metadata.find((m) => m?.name === name) || null;
  }

  // Case 3: Plain object
  if (typeof metadata === "object") {
    return metadata[name] || null;
  }

  return null;
}

function normalizeDims(dims) {
  if (!Array.isArray(dims)) {
    return null;
  }

  const normalized = dims.map((dim) => {
    if (typeof dim === "number" && Number.isInteger(dim) && dim > 0) {
      return dim;
    }

    // Some runtimes expose dims as strings
    const parsed = Number(dim);
    if (Number.isInteger(parsed) && parsed > 0) {
      return parsed;
    }

    return null;
  });

  if (normalized.some((dim) => dim === null)) {
    return null;
  }

  return normalized;
}

function historyMeta(settings, seconds) {
  const source = settings.source === "sequencer" ? "Sequencer" : "MIDI";
  const [x, y] = mapPointToModelPosition(settings.mapPoint01);
  return `${modelDisplayName(settings.modelName)} / ${seconds.toFixed(2)}s / ${source} / x ${x.toFixed(3)}, y ${y.toFixed(3)}`;
}

function addGenerationToHistory(samples, blob, settings, durationSeconds) {
  state.generationIndex += 1;
  const url = URL.createObjectURL(blob);
  const generation = {
    id: globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : String(Date.now()),
    index: state.generationIndex,
    name: `${modelDisplayName()} ${state.generationIndex}`,
    url,
    samples: new Float32Array(samples),
    peak: maxAbs(samples),
    settings,
    durationSeconds,
    audio: new Audio(url),
    canvas: null,
    playButton: null,
    progressFrame: null,
  };
  state.generations.unshift(generation);
  renderHistory();
  return generation;
}

function setIconButton(button, iconName, label) {
  if (!button) {
    return;
  }
  button.innerHTML = "";
  button.title = label;
  button.setAttribute("aria-label", label);
  button.dataset.icon = iconName;
  button.insertAdjacentHTML("beforeend", `<svg viewBox="0 0 24 24" aria-hidden="true">${iconPath(iconName)}</svg>`);
}

function iconPath(iconName) {
  const icons = {
    check: '<path d="M20 6 9 17l-5-5"/>',
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>',
    pencil: '<path d="M21.2 6.8 17.2 2.8a2.4 2.4 0 0 0-3.4 0L3 13.6V21h7.4L21.2 10.2a2.4 2.4 0 0 0 0-3.4Z"/><path d="m14 4 6 6"/>',
    play: '<path d="M8 5v14l11-7Z"/>',
    refresh: '<path d="M20 6v5h-5"/><path d="M4 18v-5h5"/><path d="M18.3 9A7 7 0 0 0 6.2 6.7L4 9"/><path d="M5.7 15A7 7 0 0 0 17.8 17.3L20 15"/>',
    "rotate-ccw": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    square: '<path d="M6 6h12v12H6z"/>',
  };
  return icons[iconName] || icons.play;
}

function downloadGeneration(generation) {
  const link = document.createElement("a");
  link.href = generation.url;
  link.download = `${sanitizeFilename(generation.name)}.wav`;
  document.body.append(link);
  link.click();
  link.remove();
}

function stopGeneration(generation, reset = true) {
  if (!generation.audio) {
    return;
  }
  generation.audio.pause();
  if (reset) {
    generation.audio.currentTime = 0;
  }
  if (generation.progressFrame) {
    cancelAnimationFrame(generation.progressFrame);
    generation.progressFrame = null;
  }
  setIconButton(generation.playButton, "play", "Play");
  drawGenerationWaveform(generation);
}

function stopOtherGenerations(activeGeneration) {
  for (const generation of state.generations) {
    if (generation !== activeGeneration) {
      stopGeneration(generation);
    }
  }
}

function updateGenerationProgress(generation) {
  drawGenerationWaveform(generation);
  if (!generation.audio.paused) {
    generation.progressFrame = requestAnimationFrame(() => updateGenerationProgress(generation));
  }
}

function toggleGenerationPlayback(generation) {
  if (!generation.audio) {
    return;
  }

  if (!generation.audio.paused) {
    stopGeneration(generation);
    return;
  }

  stopOtherGenerations(generation);
  if (generation.audio.currentTime >= generation.durationSeconds) {
    generation.audio.currentTime = 0;
  }
  generation.audio.play().then(() => {
    setIconButton(generation.playButton, "square", "Stop");
    updateGenerationProgress(generation);
  }).catch((error) => {
    appendConsole(`playback error: ${error.message || String(error)}`);
  });
}

function seekGeneration(generation, event) {
  if (!generation.audio || !generation.canvas) {
    return;
  }
  const rect = generation.canvas.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  generation.audio.currentTime = ratio * generation.durationSeconds;
  drawGenerationWaveform(generation);
}

function drawGenerationWaveform(generation) {
  if (!generation.canvas) {
    return;
  }
  const progress = generation.audio && generation.durationSeconds
    ? generation.audio.currentTime / generation.durationSeconds
    : 0;
  drawWaveform(generation.canvas, generation.samples, progress, generation.peak);
}

function maxAbs(samples) {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) {
    peak = Math.max(peak, Math.abs(samples[i]));
  }
  return peak;
}

function renderHistory() {
  el.historyList.replaceChildren();
  el.historyEmpty.hidden = state.generations.length > 0;

  for (const generation of state.generations) {
    const row = document.createElement("div");
    row.className = "history-row";
    if (generation === state.generations[0]) {
      row.classList.add("is-current");
    }

    const head = document.createElement("div");
    head.className = "history-row-head";

    const titleInput = document.createElement("input");
    titleInput.className = "history-title-input";
    titleInput.type = "text";
    titleInput.value = generation.name;
    titleInput.setAttribute("aria-label", "Generation name");
    titleInput.addEventListener("input", () => {
      generation.name = titleInput.value.trim() || `${modelDisplayName()} ${generation.index}`;
    });

    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.textContent = historyMeta(generation.settings, generation.durationSeconds);

    const playButton = document.createElement("button");
    playButton.className = "history-action";
    playButton.type = "button";
    setIconButton(playButton, generation.audio && !generation.audio.paused ? "square" : "play", generation.audio && !generation.audio.paused ? "Stop" : "Play");
    playButton.addEventListener("click", () => toggleGenerationPlayback(generation));
    generation.playButton = playButton;

    const restoreButton = document.createElement("button");
    restoreButton.className = "history-action";
    restoreButton.type = "button";
    setIconButton(restoreButton, "rotate-ccw", "Restore settings");
    restoreButton.addEventListener("click", () => restoreGeneration(generation));

    const downloadButton = document.createElement("button");
    downloadButton.className = "history-action";
    downloadButton.type = "button";
    setIconButton(downloadButton, "download", "Download audio");
    downloadButton.addEventListener("click", () => downloadGeneration(generation));

    const waveform = document.createElement("canvas");
    waveform.className = "history-waveform";
    waveform.width = 1200;
    waveform.height = 132;
    waveform.tabIndex = 0;
    waveform.setAttribute("aria-label", "Waveform seek control");
    waveform.addEventListener("click", (event) => seekGeneration(generation, event));
    generation.canvas = waveform;

    generation.audio.onended = () => {
      stopGeneration(generation);
    };

    const titleSlot = document.createElement("div");
    titleSlot.className = "history-title-slot";
    titleSlot.append(titleInput);

    const controls = document.createElement("div");
    controls.className = "history-controls";
    controls.append(playButton, downloadButton, restoreButton);

    const waveformWrap = document.createElement("div");
    waveformWrap.className = "history-waveform-wrap";
    waveformWrap.append(waveform);

    const body = document.createElement("div");
    body.className = "history-body";
    body.append(controls, waveformWrap);

    head.append(titleSlot, meta);
    row.append(head, body);
    el.historyList.append(row);
    drawGenerationWaveform(generation);
  }
}

function restoreGeneration(generation) {
  const { settings } = generation;

  if (settings.modelName && state.availableModels.some((model) => model.name === settings.modelName)) {
    state.selectedModelName = settings.modelName;
    el.modelSelect.value = settings.modelName;
    state.modelBaseUrl = getSelectedModel()?.baseUrl || state.modelBaseUrl;
  }

  el.durationSelect.value = settings.durationChunks;
  el.startTime.value = settings.startTime;
  state.mapPoint01 = { ...settings.mapPoint01 };

  const seq = settings.sequencer;
  el.sequenceNote.value = seq.sequenceNote ?? "0";
  el.sequenceOctave.value = seq.sequenceOctave ?? "3";
  el.progressionSelect.value = seq.progressionSelect;
  el.customProgression.value = seq.customProgression;
  el.chordsPerChunk.value = seq.chordsPerChunk;
  const restoredArpMode = seq.arpMode ?? "off";
  el.arpEnabled.checked = Boolean(seq.arpEnabled ?? restoredArpMode !== "off");
  el.arpMode.value = restoredArpMode === "off" ? "up" : restoredArpMode;
  el.notesPerChord.value = seq.notesPerChord ?? "3";
  el.notesArpeggiated.value = seq.notesArpeggiated ?? "2";
  el.velocityMin.value = seq.velocityMin ?? "64";
  el.velocityMax.value = seq.velocityMax ?? "115";
  el.velocityVariance.value = seq.velocityVariance ?? "0.5";
  el.velocityMinVal.textContent = Number(el.velocityMin.value);
  el.velocityMaxVal.textContent = Number(el.velocityMax.value);
  el.velocityVarianceVal.textContent = Number(el.velocityVariance.value).toFixed(2);

  state.noteSource = settings.source;
  document.querySelectorAll(".tab[data-tab]").forEach((t) => {
    t.classList.toggle("is-active", t.dataset.tab === state.noteSource);
  });
  document.getElementById("tabSequencer").hidden = state.noteSource !== "sequencer";
  document.getElementById("tabMidi").hidden = state.noteSource !== "midi";

  if (settings.source === "midi") {
    state.notes = cloneNotes(settings.midi.notes);
    state.midi = state.notes.length ? { restored: true } : null;
    el.midiMeta.textContent = settings.midi.meta || "Restored MIDI";
  }

  updateSourceUi();
  updateArpUi();
  updateCrosshair();
  drawMidiPreview();
  setStatus(`Restored ${generation.name}.`);
  appendConsole(`Restored settings from ${generation.name}`);
  refreshGenerateState();
}

function sanitizeFilename(value) {
  return (value || "after-generation")
    .replace(/[^a-z0-9._-]+/gi, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 80) || "after-generation";
}

async function generateAudio() {
  if (!state.session || !getActiveNotes().length) {
    return;
  }
  el.generateButton.disabled = true;
  setGenerateBusy(true);
  await waitForPaint();
  try {
    const start = state.noteSource === "sequencer" ? 0 : Number(el.startTime.value) || 0;
    const chunks = Number(el.durationSelect.value) || 1;
    const duration = chunks * CHUNK_SECONDS;
    const output = new Float32Array(chunks * CHUNK_SAMPLES);
    const mapTensor = makeMapTensor();
    const settings = snapshotGenerationSettings();

    const started = performance.now();
    for (let chunk = 0; chunk < chunks; chunk++) {
      setStatus(`Generating chunk ${chunk + 1} / ${chunks}...`);
      const feeds = {
        map_pos: mapTensor,
        noise: makeNoiseTensor(),
      };
      feeds[getPianoRollInputName()] = buildPianoRoll(start + chunk * CHUNK_SECONDS);
      const result = await state.session.run(feeds);
      const resultKeys = Object.keys(result);
      // appendConsole(`chunk ${chunk + 1}: outputs ${resultKeys.join(", ")}`);
      // for (const key of resultKeys) {
      //   const preview = Array.from(result[key].data.slice(0, 8))
      //     .map((value) => Number(value).toFixed(5))
      //     .join(", ");
      //   appendConsole(`${key}: dims [${result[key].dims.join(", ")}], ${preview}`);
      // }
      if (!result.audio) {
        throw new Error(`Model did not return an audio output. Outputs: ${resultKeys.join(", ")}`);
      }
      output.set(result.audio.data, chunk * CHUNK_SAMPLES);
      if (result.cond) {
        appendConsole(`cond vector: ${Array.from(result.cond.data).map((value) => Number(value).toFixed(5)).join(", ")}`);
      }

      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    const blob = floatToWavBlob(output, SAMPLE_RATE);
    addGenerationToHistory(output, blob, settings, duration);
    setStatus(`Generated ${duration}s in ${((performance.now() - started) / 1000).toFixed(2)}s.`);
  } catch (error) {
    console.error(error);
    appendConsole(`error: ${error.message || String(error)}`);
    setStatus(error.message || String(error), true);
  } finally {
    setGenerateBusy(false);
    refreshGenerateState();
  }
}

function drawWaveform(canvas, samples, progress = 0, peak = null) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#eceff2";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#111318";
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  const mid = height / 2;
  const step = Math.max(1, Math.floor(samples.length / width));
  const normalizedPeak = peak ?? maxAbs(samples);
  const scale = normalizedPeak > 1e-7 ? 1 / normalizedPeak : 1;
  for (let x = 0; x < width; x++) {
    const start = x * step;
    let min = 0;
    let max = 0;
    for (let i = 0; i < step && start + i < samples.length; i++) {
      const v = samples[start + i] * scale;
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
    ctx.moveTo(x, mid + min * mid * 0.88);
    ctx.lineTo(x, mid + max * mid * 0.88);
  }
  ctx.stroke();

  const clampedProgress = Math.max(0, Math.min(1, progress || 0));
  if (clampedProgress > 0) {
    const x = clampedProgress * width;
    ctx.fillStyle = "rgba(11, 111, 99, 0.12)";
    ctx.fillRect(0, 0, x, height);
    ctx.strokeStyle = "#0b6f63";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
}

function floatToWavBlob(samples, sampleRate) {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view, offset, value) {
  for (let i = 0; i < value.length; i++) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

async function loadDefaultMapImage() {
  const response = await fetch(DEFAULT_MAP_IMAGE_URL, { cache: "force-cache" });

  if (!response.ok) {
    throw new Error(`Could not load default map image: HTTP ${response.status}`);
  }

  const blob = await response.blob();
  const img = new Image();

  if (state.mapObjectUrl) {
    URL.revokeObjectURL(state.mapObjectUrl);
  }

  state.mapObjectUrl = URL.createObjectURL(blob);
  img.src = state.mapObjectUrl;
  await img.decode();

  state.mapImage = img;
  drawMap();
}

function drawMap() {
  state._drawMapCallCount++;
  const stack = new Error().stack?.split('\n').slice(1, 3).join(' | ') || '';
  appendConsole(`[${state._drawMapCallCount}] drawMap: mapImage=${!!state.mapImage}, from: ${stack}`);
  const ctx = el.mapCanvas.getContext("2d");
  const cssSize = Math.round(el.mapWrap.getBoundingClientRect().width || 900);
  if (el.mapCanvas.width !== cssSize || el.mapCanvas.height !== cssSize) {
    el.mapCanvas.width = cssSize;
    el.mapCanvas.height = cssSize;
  }
  const { width, height } = el.mapCanvas;


  if (!state.mapImage) {
    appendConsole("drawMap: state.mapImage is empty, loading default");
    loadDefaultMapImage().catch((error) => {
      appendConsole(`default map image failed: ${error.message || String(error)}`);
    });
    return;
  }
  
  ctx.clearRect(0, 0, width, height);
  // ctx.fillStyle = "#eef1f4";
  // ctx.fillRect(0, 0, width, height);
  ctx.drawImage(state.mapImage, 0, 0, width, height);
}

function updateCrosshair() {
  el.crosshair.style.left = `${state.mapPoint01.x * 100}%`;
  el.crosshair.style.top = `${state.mapPoint01.y * 100}%`;
  const [x, y] = getMapPosition();
  el.coords.textContent = `x: ${x.toFixed(4)}, y: ${y.toFixed(4)}`;
}

function getMapPosition() {
  return mapPointToModelPosition(state.mapPoint01);
}

function mapPointToModelPosition(point01) {
  const x = (point01.x * 2 - 1) * MAP_RANGE;
  const y = (1 - point01.y * 2) * MAP_RANGE;
  return [x, y];
}

function initTabs() {
  document.querySelectorAll(".tab[data-tab]").forEach((tab) => {
    tab.addEventListener("click", () => {
      const tabId = tab.dataset.tab;
      document.querySelectorAll(".tab[data-tab]").forEach((t) => t.classList.toggle("is-active", t === tab));
      document.getElementById("tabSequencer").hidden = tabId !== "sequencer";
      document.getElementById("tabMidi").hidden = tabId !== "midi";
      state.noteSource = tabId === "sequencer" ? "sequencer" : "midi";
      updateSourceUi();
      drawMidiPreview();
      refreshGenerateState();
    });
  });
}

function initVelocitySliders() {
  el.velocityMin.addEventListener("input", () => {
    if (Number(el.velocityMin.value) > Number(el.velocityMax.value)) {
      el.velocityMin.value = el.velocityMax.value;
    }
    el.velocityMinVal.textContent = Number(el.velocityMin.value);
    drawMidiPreview();
  });
  el.velocityMax.addEventListener("input", () => {
    if (Number(el.velocityMax.value) < Number(el.velocityMin.value)) {
      el.velocityMax.value = el.velocityMin.value;
    }
    el.velocityMaxVal.textContent = Number(el.velocityMax.value);
    drawMidiPreview();
  });
  el.velocityVariance.addEventListener("input", () => {
    el.velocityVarianceVal.textContent = Number(el.velocityVariance.value).toFixed(2);
    drawMidiPreview();
  });
}

function rerollVelocities() {
  drawMidiPreview();
  refreshGenerateState();
}

function rerollArpPattern() {
  drawMidiPreview();
  refreshGenerateState();
}

el.midiFile.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) {
    loadMidiFile(file).catch((error) => setStatus(error.message || String(error), true));
  }
});

el.durationSelect.addEventListener("change", () => { updateSourceUi(); drawMidiPreview(); refreshGenerateState(); });
el.startTime.addEventListener("input", drawMidiPreview);
el.generateButton.addEventListener("click", generateAudio);
el.rerollVelocityButton.addEventListener("click", rerollVelocities);
el.rerollArpButton.addEventListener("click", rerollArpPattern);

function updateArpUi() {
  const enabled = el.arpEnabled.checked;
  el.arpMode.disabled = !enabled;
  el.notesArpeggiated.disabled = !enabled;
  el.rerollArpButton.disabled = !(enabled && el.arpMode.value === "random");
}

[
  el.sequenceNote,
  el.sequenceOctave,
  el.progressionSelect,
  el.customProgression,
  el.chordsPerChunk,
  el.arpEnabled,
  el.arpMode,
  el.notesPerChord,
  el.notesArpeggiated,
].forEach((control) => {
  const updateHandler = () => {
    updateSourceUi();
    updateArpUi();
    drawMidiPreview();
    refreshGenerateState();
  };
  control.addEventListener("input", updateHandler);
  control.addEventListener("change", updateHandler);
});

function updateSourceUi() {
  if (state.noteSource === "sequencer") {
    const chunks = Number(el.durationSelect.value) || 1;
    const chords = chunks * Math.max(1, Math.floor(Number(el.chordsPerChunk.value) || 1));
    el.midiMeta.textContent = `Sequencer | ${chords} chords | ${getProgressionTokens().join(" ")}`;
  } else if (!state.midi) {
    el.midiMeta.textContent = "No MIDI loaded";
  }
}

function updateMidiTooltip(event) {
  const canvas = el.midiCanvas;
  const tooltip = document.getElementById("midiTooltip");

  if (!state.midiPreviewData || !state.midiPreviewData.visible.length) {
    tooltip.hidden = true;
    return;
  }

  const rect = canvas.getBoundingClientRect();
  const cssX = event.clientX - rect.left;
  const cssY = event.clientY - rect.top;
  const mouseX = cssX * (canvas.width / rect.width);
  const mouseY = cssY * (canvas.height / rect.height);

  if (mouseX < 0 || mouseX > canvas.width || mouseY < 0 || mouseY > canvas.height) {
    tooltip.hidden = true;
    return;
  }

  const { width, height } = canvas;
  const { start, duration, visible, minPitch, maxPitch } = state.midiPreviewData;
  const plotW = width - ROLL_LEFT;
  const N = maxPitch - minPitch + 1;
  const rowH = height / N;

  let foundNote = null;
  for (const note of visible) {
    const x0 = ROLL_LEFT + Math.max(0, (note.time - start) / duration) * plotW;
    const x1 = ROLL_LEFT + Math.min(1, (note.time + note.duration - start) / duration) * plotW;
    const i = note.midi - minPitch;
    const yTop = height - (i + 1) * rowH;

    if (mouseX >= x0 && mouseX <= x1 && mouseY >= yTop && mouseY <= yTop + rowH) {
      foundNote = note;
      break;
    }
  }

  if (foundNote) {
    const octNum = Math.floor(foundNote.midi / 12) - 1;
    const noteName = NOTE_NAMES[foundNote.midi % 12];
    const velocity127 = Math.round(foundNote.velocity * 127);
    tooltip.textContent = `${noteName}${octNum} vel:${velocity127}`;
    tooltip.style.left = `${Math.min(cssX + 10, rect.width - 100)}px`;
    tooltip.style.top = `${cssY - 20}px`;
    tooltip.hidden = false;
  } else {
    tooltip.hidden = true;
  }
}

el.mapWrap.addEventListener("pointerdown", (event) => {
  const rect = el.mapWrap.getBoundingClientRect();
  state.mapPoint01.x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  state.mapPoint01.y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
  updateCrosshair();
});

el.midiCanvas.addEventListener("mousemove", updateMidiTooltip);
el.midiCanvas.addEventListener("mouseleave", () => {
  const tooltip = document.getElementById("midiTooltip");
  tooltip.hidden = true;
});

initTabs();
initVelocitySliders();
setIconButton(el.rerollVelocityButton, "refresh", "Reroll velocities");
setIconButton(el.rerollArpButton, "refresh", "Reroll arpegiation");
updateArpUi();
drawMap();
for (const option of el.durationSelect.options) {
  const chunks = Number(option.value);
  option.textContent = `${chunks} chunk${chunks > 1 ? "s" : ""} (${(chunks * CHUNK_SECONDS).toFixed(2)} s)`;
}
updateSourceUi();
drawMidiPreview();
updateCrosshair();
appendConsole("Ready.");
scanModels().catch((error) => {
  console.error(error);
  // el.modelStatus.textContent = "Model scan failed";
  setStatus(error.message || String(error), true);
});
