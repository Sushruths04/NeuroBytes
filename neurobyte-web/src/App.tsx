import { useTheme } from "./lib/themeMode";
import { ThemeToggle } from "./components/ThemeToggle";
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchManifest, fetchSample, type Manifest } from "./lib/api";
import { useModelChoice } from "./lib/modelChoice";
import { ModelPicker } from "./components/ModelPicker";
import { modelInfo, type ModelId } from "./content/models";
import { ParticleBrain } from "./components/ParticleBrain";
import { channelActivity, loadRecording, type Recording } from "./lib/eeg";
import { matchRecording, runPipeline, stageAt, TOTAL_SECONDS, type MatchOutcome, type PipelineResult } from "./lib/pipeline";
import { href, useRoute, type Route } from "./lib/router";
import { clearSession, loadSession, saveSession } from "./lib/session";
import { ScalpHeatmap } from "./components/ScalpHeatmap";
import { EegCanvas } from "./components/EegCanvas";
import { ResultsView } from "./components/ResultsView";
import { RunView } from "./components/RunView";
import { ScienceView } from "./components/ScienceView";
import { AnalyticsSections } from "./components/AnalyticsView";
import { SubjectPicker } from "./components/SubjectPicker";
import { useNeuralTransition } from "./components/NeuralTransition";

type Phase = "idle" | "reading" | "running" | "done" | "unmatched";

export default function App() {
  const [route, go] = useRoute();
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [rec, setRec] = useState<Recording | null>(null);
  const [match, setMatch] = useState<MatchOutcome | null>(null);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [saved, setSaved] = useState<string | null>(null); // file name of a resumable session
  const [restoring, setRestoring] = useState(false);
  const start = useRef(0);
  const runId = useRef(0);
  const transition = useNeuralTransition();
  const [model, setModel] = useModelChoice();

  const modelRef = useRef(model);
  modelRef.current = model;
  const [manifest, setManifest] = useState<Manifest | null>(null);
  useEffect(() => {
    fetchManifest().then(setManifest).catch(() => setManifest(null));
  }, []);

  /** Drop the current analysis and return to the upload page. */
  const uploadAnother = useCallback(async () => {
    runId.current++;
    setPhase("idle");
    setRec(null);
    setMatch(null);
    setResult(null);
    setElapsed(0);
    setError(null);
    setSaved(null);
    // Wait for the delete to commit before navigating, otherwise an immediate
    // visit to #/analysis could still find (and restore) the old session.
    await clearSession();
    go("home");
  }, [go]);

  const begin = useCallback(
    async (buf: ArrayBuffer, name: string, opts: { restored?: boolean } = {}) => {
      const id = ++runId.current;
      setError(null);
      setResult(null);
      setMatch(null);
      setPhase("reading");
      try {
        const r = await loadRecording(buf, name);
        if (id !== runId.current) return;
        setRec(r);
        if (!opts.restored) {
          await saveSession(name, buf);
          setSaved(name);
        }
        // Restored sessions and reduced-motion users skip the guided animation.
        const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
        start.current = performance.now() - (reduce || opts.restored ? TOTAL_SECONDS * 1000 : 0);
        setElapsed(0);
        setPhase("running");
        go("analysis");
        const mdl = modelRef.current;
        const m = await matchRecording(r, mdl);
        if (id !== runId.current) return;
        setMatch(m);
        if (m.ok) {
          const res = await runPipeline(m.subject, m.manifest, mdl);
          if (id === runId.current) setResult(res);
        }
      } catch (e) {
        if (id !== runId.current) return;
        setError(e instanceof Error ? e.message : String(e));
        setPhase("idle");
        setRec(null);
        if (opts.restored) void clearSession();
        go("home");
      }
    },
    [go],
  );

  const sample = useCallback(
    async (sub: string) => {
      try {
        setError(null);
        setPhase("reading");
        await begin(await fetchSample(sub), `${sub}_tst.npy`);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setPhase("idle");
      }
    },
    [begin],
  );

  // Switching model re-ranks the recording that is open, without replaying the animation.
  const lastModel = useRef(model);
  useEffect(() => {
    if (lastModel.current === model) return;
    lastModel.current = model;
    if (!rec || (phase !== "done" && phase !== "unmatched" && phase !== "running")) return;
    const id = ++runId.current;
    setResult(null);
    void (async () => {
      try {
        const m = await matchRecording(rec, model);
        if (id !== runId.current) return;
        setMatch(m);
        if (!m.ok) return setPhase((p) => (p === "running" ? p : "unmatched"));
        const res = await runPipeline(m.subject, m.manifest, model);
        if (id !== runId.current) return;
        setResult(res);
        setPhase((p) => (p === "unmatched" ? "done" : p));
      } catch (e) {
        if (id === runId.current) setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, [model, rec, phase]);

  /** Results view: jump to another person's test recording (no animation). */
  /**
   * Results view: switch to another person's test recording without leaving the
   * results. Everything loads in the background while the current results stay on
   * screen; recording, match and ranking are then swapped in one render.
   */
  const [switching, setSwitching] = useState<string | null>(null);
  const [switchError, setSwitchError] = useState<string | null>(null);
  const pickSubject = useCallback(async (sub: string) => {
    const id = ++runId.current;
    const name = `${sub}_tst.npy`;
    setSwitching(sub);
    setSwitchError(null);
    try {
      const buf = await fetchSample(sub);
      const r = await loadRecording(buf, name);
      const mdl = modelRef.current;
      const m = await matchRecording(r, mdl);
      if (!m.ok) throw new Error(m.reason);
      const res = await runPipeline(m.subject, m.manifest, mdl);
      if (id !== runId.current) return; // a newer switch or upload took over
      await saveSession(name, buf);
      setSaved(name);
      setRec(r);
      setMatch(m);
      setResult(res);
      setPhase("done");
    } catch (e) {
      if (id === runId.current) setSwitchError(e instanceof Error ? e.message : String(e));
    } finally {
      if (id === runId.current) setSwitching(null);
    }
  }, []);

  // Whenever we land on #/analysis without a recording in memory (a reload, or
  // navigating back after a reload elsewhere), restore the saved session so the
  // user stays on the person they last chose. With nothing saved (first visit via
  // the Analysis tab, or after "Choose another person"), open sub-01.
  const restoringRef = useRef(false);
  useEffect(() => {
    if (route !== "analysis" || rec || restoringRef.current || phase === "reading") return;
    restoringRef.current = true;
    setRestoring(true);
    void (async () => {
      try {
        const s = await loadSession();
        if (s) await begin(s.buffer, s.fileName, { restored: true });
        else await begin(await fetchSample(DEFAULT_SUBJECT), `${DEFAULT_SUBJECT}_tst.npy`);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        go("home");
      }
      restoringRef.current = false;
      setRestoring(false);
    })();
  }, [route, rec, phase, go, begin]);

  // Show the "resume" card on the home page after a reload.
  useEffect(() => {
    void loadSession().then((s) => s && setSaved((cur) => cur ?? s.fileName));
  }, []);

  const resume = useCallback(async () => {
    if (rec) return go("analysis"); // still in memory
    const s = await loadSession();
    if (s) await begin(s.buffer, s.fileName, { restored: true });
  }, [rec, go, begin]);

  // A file dropped outside the drop zone would make the browser navigate away
  // to it, which looks like the app "refreshing". Swallow those drops.
  useEffect(() => {
    const stop = (e: DragEvent) => {
      if (e.dataTransfer?.types.includes("Files")) e.preventDefault();
    };
    window.addEventListener("dragover", stop);
    window.addEventListener("drop", stop);
    return () => {
      window.removeEventListener("dragover", stop);
      window.removeEventListener("drop", stop);
    };
  }, []);

  // animation clock
  useEffect(() => {
    if (phase !== "running") return;
    let raf = 0;
    const tick = () => {
      setElapsed((performance.now() - start.current) / 1000);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [phase]);

  const stage = stageAt(elapsed);

  // phase transitions driven by the clock + async results
  useEffect(() => {
    if (phase !== "running") return;
    if (match && !match.ok && stage.index >= 2) setPhase("unmatched");
    else if (result && stage.index >= 8) setPhase("done");
  }, [phase, match, result, stage.index]);

  const skip = () => {
    start.current = performance.now() - TOTAL_SECONDS * 1000;
  };

  // Don't animate past "scorers" until the score matrix has actually loaded.
  const shownStage = !result && stage.index >= 4 ? { index: 4, progress: 0.99 } : stage;
  const hasAnalysis = !!rec || !!saved;

  return (
    <>
    <ParticleBrain />
    <div className="relative z-10 mx-auto max-w-[1400px] px-4 pb-16 sm:px-6">
      <Header
        route={route}
        subject={match?.ok ? match.subject : null}
        model={model}
        onModel={setModel}
        available={availableModels(manifest, match?.ok ? match.subject : null)}
        go={go}
      />

      <div data-page-content>
      {route === "science" && <ScienceView />}


      {route === "home" && (
        <main className="grid gap-24 pb-12 pt-10">
          <section className="grid items-center gap-10 lg:gap-0 lg:grid-cols-[minmax(0,1fr)_var(--spine-gutter)_minmax(0,1fr)]">
            <Hero model={model} />
            {/* the fixed particle canvas draws the brain over this box */}
            <div data-brain-anchor className="relative h-72 sm:h-[26rem] lg:col-start-3" aria-hidden />
          </section>

          <section className="grid gap-6 lg:gap-0 lg:grid-cols-[minmax(0,1fr)_var(--spine-gutter)_minmax(0,1fr)]">
            <SubjectPicker busy={phase === "reading"} error={error} onSelect={(sub, el) => transition(() => sample(sub), el, { style: "slide" })} current={match?.ok ? match.subject : null} />
            {hasAnalysis && phase !== "reading" && (
              <div className="self-end rounded-3xl border border-highlight-med p-5 lg:col-start-3" aria-label="Resume analysis" role="region">
                <p className="eyebrow">In progress</p>
                <p className="mt-2 font-mono text-sm text-text">{rec?.fileName ?? saved}</p>
                <div className="mt-4 flex gap-2">
                  <button type="button" className="btn-primary" onClick={() => void resume()}>Back to results</button>
                  <button type="button" className="btn-ghost" onClick={uploadAnother}>Discard</button>
                </div>
              </div>
            )}
          </section>

          <HomeExtras />
          <AnalyticsSections model={model} />
        </main>
      )}

      {route === "analysis" && (
        <>
          {(!rec || phase === "reading") && (
            <main className="pt-16 text-center text-sm text-subtle" role="status">
              {restoring ? "Restoring your analysis…" : "Loading…"}
            </main>
          )}

          {phase === "running" && rec && (
            <main className="pt-6">
              <AnalysisBar fileName={rec.fileName} onUploadAnother={uploadAnother} />
              <RunView
                model={model}
                rec={rec}
                subject={match?.ok ? match.subject : null}
                elapsed={elapsed}
                stage={shownStage}
                result={result}
                onSkip={skip}
              />
            </main>
          )}

          {phase === "done" && rec && !result && (
            <main className="pt-16 text-center text-sm text-subtle" role="status">Ranking with {model}…</main>
          )}

          {phase === "done" && rec && result && (
            <main className="pt-6">
              <ResultsView
                rec={rec}
                result={result}
                onReset={uploadAnother}
                onPickSubject={(s) => void pickSubject(s)}
                switching={switching}
                switchError={switchError}
              />
            </main>
          )}

          {phase === "unmatched" && rec && match && !match.ok && (
            <Unmatched rec={rec} reason={match.reason} onReset={uploadAnother} />
          )}
        </>
      )}
      </div>
    </div>
    </>
  );
}

/** Opened when the Analysis tab is used before any person has been chosen. */
const DEFAULT_SUBJECT = "sub-01";

function AnalysisBar({ fileName, onUploadAnother }: { fileName: string; onUploadAnother: () => void }) {
  return (
    <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
      <p className="text-xs text-subtle">
        Analysing <span className="font-mono text-text">{fileName}</span> · saved in this browser, so a refresh keeps your place
      </p>
      <button type="button" className="btn-ghost px-3 py-1.5 text-xs" onClick={onUploadAnother}>
        ← Choose another person
      </button>
    </div>
  );
}

/** Models with a score matrix for this subject (all, when no subject is open). */
function availableModels(manifest: Manifest | null, subject: string | null): string[] | null {
  if (!manifest?.models) return null;
  return Object.entries(manifest.models)
    .filter(([, subs]) => (subject ? subs.includes(subject) : subs.length > 0))
    .map(([m]) => m);
}

interface HeaderProps {
  route: Route;
  subject: string | null;
  model: ModelId;
  onModel: (m: ModelId) => void;
  available: string[] | null;
  go: (r: Route) => void;
}

function Header({ route, subject, model, onModel, available, go }: HeaderProps) {
  const { t } = useTheme();
  const transition = useNeuralTransition();
  // nav links play the neuron transition; the href stays for new-tab / no-JS use
  const nav = (r: Route) => (e: React.MouseEvent<HTMLAnchorElement>) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    if (r !== route) transition(() => go(r), e.currentTarget);
  };
  const link = (r: Route, label: string) => (
    <a
      href={href(r)}
      onClick={nav(r)}
      aria-current={route === r ? "page" : undefined}
      className={`px-3 py-1.5 text-[13px] font-semibold uppercase tracking-[0.06em] transition ${
        route === r ? "text-text" : "text-muted hover:text-text"
      }`}
    >
      {label}
    </a>
  );
  return (
    <header className="flex flex-wrap items-center justify-between gap-3 py-5">
      <a href={href("home")} onClick={nav("home")} className="flex items-center gap-2.5" aria-label="NeuroBytes home">
        <svg viewBox="0 0 32 32" className="h-8 w-8" aria-hidden>
          <path d="M16 3 29 27H3Z" fill={t.iris} />
          <path d="M7 21h4l2-4 2.5 6 3-10 2 8h4.5" fill="none" stroke={t["on-iris"]} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="text-[15px] font-semibold tracking-[0.12em] text-text">NEUROBYTES</span>
      </a>
      <nav aria-label="Main" className="flex items-center gap-1">
        {link("home", "Home")}
        {link("analysis", "Analysis")}
        {link("science", "Why it works")}
      </nav>
      <div className="flex min-w-0 items-center gap-2">
        {subject && <span className="chip hidden bg-overlay text-foam sm:inline-flex">{subject}</span>}
        <ModelPicker model={model}
 onChange={onModel} available={available} compact />
        <ThemeToggle />
      </div>
    </header>
  );
}

/** Landing hero: headline numbers of the selected model, particle brain, how it works. */
function Hero({ model }: { model: ModelId }) {
  const m = modelInfo(model);
  return (
    <div>
      <p className="eyebrow">EEG-to-video retrieval</p>
      <h1 className="mt-5 text-5xl leading-[1.05] sm:text-7xl">
        See what the brain <span className="grad-word">watched.</span>
      </h1>
      <p className="lead mt-6 max-w-xl">
        From three seconds of EEG over the visual cortex, NeuroBytes ranks 102 candidate videos and finds the one the
        person was watching. No video is shown to the model at test time, only brain activity.
      </p>
      <dl className="mt-10 grid max-w-xl grid-cols-2 gap-x-8 gap-y-6 sm:grid-cols-4" aria-label={`${m.name} dev-test results`}>
        {[
          [`${m.frozen.top1.toFixed(2)}%`, "top-1 · chance 0.98%"],
          [`${m.frozen.top5.toFixed(2)}%`, "top-5 · chance 4.9%"],
          [m.frozen.meanRank.toFixed(2), "mean rank of 102"],
          [`×${Math.round(m.frozen.top1 / 0.98)}`, "better than chance"],
        ].map(([v, l]) => (
          <div key={l}>
            <dd className="text-3xl tracking-tight text-text">{v}</dd>
            <dt className="mt-1 text-[11px] uppercase tracking-[0.08em] text-muted">{l}</dt>
          </div>
        ))}
      </dl>
      <p className="mt-4 text-xs font-light text-muted">{m.name} – Tested on sub-01 to sub-04</p>
      <div className="mt-8 flex flex-wrap items-center gap-6">
        <button
          type="button"
          className="btn-primary"
          onClick={() => document.getElementById("pick-title")?.scrollIntoView({ behavior: "smooth", block: "center" })}
        >
          Choose a person
        </button>
        <a href={href("science")} className="text-sm text-text underline decoration-highlight-high underline-offset-4 hover:decoration-gold">
          Why it works →
        </a>
      </div>
    </div>
  );
}

const FEATURES = [
  { title: "EEG Moments data", text: "6 people, 17 channels over the back of the head, ~1,000 training clips each, 102 test clips averaged over repeats." },
  { title: "~590 brain-to-video scorers", text: "Ridge decoders over latency windows × deep video features, plus two contrastive neural networks trained across people." },
  { title: "Cognitive grounding", text: "The decodable signal sits in the first 800 ms, the classic visual ERP window. We tested 63 brain-defined attributes against it." },
  { title: "Validated on held-out data", text: "Every weight is chosen on held-out training videos; the development labels are used only to report results and compare the final candidates." },
];

const STEPS = [
  ["Record", "3 s of EEG while a clip plays"],
  ["Window", "0–700 ms after the clip starts"],
  ["Decode", "EEG → V-JEPA, CLIP, DINOv2, audio…"],
  ["Fuse", "rank fusion of ~590 scorers"],
  ["Balance", "CSLS hub removal + Sinkhorn"],
];

function HomeExtras() {
  return (
    <section className="grid gap-12 lg:gap-0 lg:grid-cols-[minmax(0,1fr)_var(--spine-gutter)_minmax(0,1fr)]" aria-labelledby="how-title">
      <div>
        <p className="eyebrow">How it works</p>
        <h2 id="how-title" className="mt-4 text-4xl sm:text-5xl">Three seconds, <span className="grad-word">five steps.</span></h2>
        <ol className="mt-8 grid gap-5">
          {STEPS.map(([t, d], i) => (
            <li key={t} className="flex items-baseline gap-4">
              <span className="font-mono text-sm text-iris">{String(i + 1).padStart(2, "0")}</span>
              <span>
                <span className="block text-xl text-text">{t}</span>
                <span className="block text-sm font-light text-subtle">{d}</span>
              </span>
            </li>
          ))}
        </ol>
      </div>
      <div className="grid content-start gap-x-10 gap-y-8 sm:grid-cols-2 lg:col-start-3" aria-label="Features">
        {FEATURES.map((f) => (
          <article key={f.title}>
            <h3 className="text-xl text-text">{f.title}</h3>
            <p className="mt-2 text-sm font-light leading-relaxed text-subtle">{f.text}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function Unmatched({ rec, reason, onReset }: { rec: Recording; reason: string; onReset: () => void }) {
  const [trial, setTrial] = useState(0);
  return (
    <main className="grid gap-5 pt-6">
      <section role="alert" className="card border-gold/50 p-4">
        <h2 className="card-title text-gold">Can't rank this recording in the browser</h2>
        <p className="mt-1 text-sm text-subtle">{reason}</p>
        <p className="mt-1 font-mono text-[11px] text-subtle">sha256 {rec.sha256}</p>
        <button type="button" className="btn-primary mt-3" onClick={onReset}>Try another file</button>
      </section>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
        <section className="card p-4">
          <div className="mb-2 flex items-center justify-between gap-3">
            <h2 className="card-title">{rec.fileName} · trial {trial + 1} / {rec.nTrials}</h2>
            <input
              type="range" min={0} max={rec.nTrials - 1} value={trial}
              onChange={(e) => setTrial(Number(e.target.value))}
              aria-label="Trial" className="w-48 accent-iris"
            />
          </div>
          <EegCanvas rec={rec} trial={trial} cursor={null} />
        </section>
        <section className="card p-4">
          <h2 className="card-title">Scalp heatmap (0.2 s after onset)</h2>
          <ScalpHeatmap activity={channelActivity(rec, trial, 0.2)} label="Signal strength (RMS) per electrode at 0.2 s, relative to the other channels." />
        </section>
      </div>
    </main>
  );
}
