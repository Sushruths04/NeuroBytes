/**
 * Persists the current analysis so a page reload (dev-server restart, HMR full
 * reload, accidental refresh) returns the user to where they were instead of
 * the landing page. The recording stays on this machine: IndexedDB only.
 */
const DB = "neurobytes";
const STORE = "session";
const KEY = "current";

export interface SavedSession {
  fileName: string;
  buffer: ArrayBuffer;
  savedAt: number;
}

export interface ViewState {
  trial: number;
  view: "post" | "pre";
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  return new Promise<T>((resolve, reject) => {
    const r = fn(db.transaction(STORE, mode).objectStore(STORE));
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  }).finally(() => db.close());
}

export async function saveSession(fileName: string, buffer: ArrayBuffer): Promise<void> {
  try {
    await tx("readwrite", (s) => s.put({ fileName, buffer, savedAt: Date.now() } satisfies SavedSession, KEY));
  } catch {
    /* storage unavailable (private mode, quota): the app still works, just not across reloads */
  }
}

export async function loadSession(): Promise<SavedSession | null> {
  try {
    return ((await tx("readonly", (s) => s.get(KEY))) as SavedSession | undefined) ?? null;
  } catch {
    return null;
  }
}

export async function clearSession(): Promise<void> {
  sessionStorage.removeItem(VIEW_KEY);
  try {
    await tx("readwrite", (s) => s.delete(KEY));
  } catch {
    /* ignore */
  }
}

const VIEW_KEY = "neurobytes:view";

export function saveView(v: ViewState) {
  sessionStorage.setItem(VIEW_KEY, JSON.stringify(v));
}

export function loadView(): ViewState {
  try {
    const v = JSON.parse(sessionStorage.getItem(VIEW_KEY) ?? "");
    if (Number.isInteger(v.trial) && v.trial >= 0 && (v.view === "post" || v.view === "pre")) return v;
  } catch {
    /* fall through */
  }
  return { trial: 0, view: "post" };
}
