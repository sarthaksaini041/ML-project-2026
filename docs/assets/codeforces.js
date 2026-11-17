import {normalizeRatings, normalizeSubmission} from "./features.js?v=19";

const API = "https://codeforces.com/api/";
const SPACING = 2150;
const CACHE_TTL = 15 * 60 * 1000;
const PAGE_SIZE = 1000;
const memory = new Map();
let lastRequest = 0;

function abortableDelay(ms, signal) {
  signal.throwIfAborted();
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", cancel);
      resolve();
    }, ms);
    function cancel() {
      clearTimeout(timer);
      reject(signal.reason);
    }
    signal.addEventListener("abort", cancel, {once: true});
  });
}

async function pace(signal) {
  let shared = 0;
  try { shared = Number(localStorage.getItem("skillgap:last-request")) || 0; } catch {}
  await abortableDelay(Math.max(0, Math.max(shared, lastRequest) + SPACING - Date.now()), signal);
  lastRequest = Date.now();
  try { localStorage.setItem("skillgap:last-request", String(lastRequest)); } catch {}
}

async function request(method, params, signal) {
  const url = new URL(method, API);
  url.search = new URLSearchParams(params).toString();
  for (let attempt = 0; attempt < 3; attempt++) {
    async function send() {
      await pace(signal);
      const timeout = AbortSignal.timeout(24000);
      try {
        return await fetch(url, {credentials: "omit", signal: AbortSignal.any([signal, timeout])});
      } catch (error) {
        signal.throwIfAborted();
        if (timeout.aborted) throw new Error("Codeforces took too long to respond. Try again.");
        throw new Error("Could not reach Codeforces. Check your connection or try the example.");
      }
    }
    const response = navigator.locks
      ? await navigator.locks.request("skillgap-codeforces", {signal}, send)
      : await send();
    const payload = await response.json().catch(() => null);
    const comment = payload?.comment ?? "";
    if (response.status === 429 || response.status >= 500 || /limit exceeded/i.test(comment)) {
      if (attempt < 2) {
        await abortableDelay(2200 * (attempt + 1), signal);
        continue;
      }
      throw new Error("Codeforces is busy or rate limiting requests. Please try again shortly.");
    }
    if (response.ok && payload?.status === "OK") return payload.result;
    if (method === "user.info" && /not found|handle.*invalid/i.test(comment)) {
      throw new Error("Codeforces could not find that handle. Check the spelling and try again.");
    }
    throw new Error("Codeforces could not complete this lookup. " + (comment || "Please try again."));
  }
}

export async function loadHistory(handle, signal, progress) {
  const key = handle.toLowerCase();
  const cached = memory.get(key);
  if (cached && Date.now() - cached.fetchedAt < CACHE_TTL) return cached;

  progress("Checking the Codeforces handle…");
  const [account] = await request("user.info", {handles: handle}, signal);
  if (!account?.handle) throw new Error("Codeforces did not return an account for that handle.");
  const fetchedAt = Date.now();
  progress("Loading rating history…");
  const ratings = await request("user.rating", {handle: account.handle}, signal);
  const collected = new Map();
  for (let offset = 1; ; offset += PAGE_SIZE) {
    signal.throwIfAborted();
    progress(collected.size ? "Read " + collected.size.toLocaleString() + " submissions. Checking for earlier attempts…"
      : "Loading submission history…");
    const page = await request("user.status", {handle: account.handle, from: offset, count: PAGE_SIZE}, signal);
    if (!Array.isArray(page)) throw new Error("Codeforces returned an incomplete submission history.");
    const before = collected.size;
    for (const row of page) {
      if (Number.isFinite(row.id) && Number.isFinite(row.creationTimeSeconds)) {
        collected.set(row.id, normalizeSubmission(row));
      }
    }
    if (page.length < PAGE_SIZE) break;
    if (collected.size === before) {
      throw new Error("Submission pages stopped advancing. Please retry so the full history can be checked.");
    }
  }
  const result = {
    handle: account.handle, ratings: normalizeRatings(ratings), history: [...collected.values()], fetchedAt,
  };
  memory.set(key, result);
  memory.set(account.handle.toLowerCase(), result);
  return result;
}

let catalogCache = null;

export async function loadLiveCatalog(signal) {
  if (catalogCache) return catalogCache;
  const response = await request("problemset.problems", {lang: "en"}, signal);
  if (!Array.isArray(response?.problems)) throw new Error("Codeforces did not return a problem catalog.");
  catalogCache = Object.fromEntries(response.problems
    .filter(p => Number.isInteger(p.contestId) && p.index)
    .filter(p => p.type === "PROGRAMMING")
    .map(p => [p.contestId + ":" + p.index, [
      p.name, p.rating ?? null,
      Array.isArray(p.tags) ? [...new Set(p.tags.map(tag => tag.trim().toLowerCase()))].sort() : [],
    ]]));
  return catalogCache;
}
