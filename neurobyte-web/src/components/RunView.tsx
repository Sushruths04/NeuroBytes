import { useMemo } from "react";
import { channelActivity, EPOCH_S, type Recording } from "../lib/eeg";
import { STAGES, stagesFor, type PipelineResult } from "../lib/pipeline";
import { modelInfo, type ModelId } from "../content/models";
import { ScalpHeatmap } from "./ScalpHeatmap";
import { EegCanvas } from "./EegCanvas";
import { PipelineStages } from "./PipelineStages";
import { VideoCard } from "./VideoCard";

interface Props {
  rec: Recording;
  subject: string | null;
  elapsed: number;
  stage: { index: number; progress: number };
  result: PipelineResult | null;
  model: ModelId;
  onSkip: () => void;
}

const ANALYSIS_START = STAGES.slice(0, 2).reduce((a, s) => a + s.seconds, 0);
const SCORERS = STAGES.findIndex((s) => s.id === "scorers");
const SINKHORN = STAGES.findIndex((s) => s.id === "sinkhorn");

/**
 * The guided flow while the pipeline runs: one trial per 3 s (the epoch length),
 * EEG cursor sweeping in real time, brain nodes lit by that channel's activity
 * under the cursor, and the video the pipeline pairs with this trial.
 */
export function RunView({ rec, subject, elapsed, stage, result, model, onSkip }: Props) {
  const stages = stagesFor(model);
  const info = modelInfo(model);
  const tA = Math.max(0, elapsed - ANALYSIS_START);
  const trial = Math.floor(tA / EPOCH_S) % rec.nTrials;
  const cursor = stage.index < 2 ? null : tA % EPOCH_S;
  const activity = useMemo(
    () => channelActivity(rec, trial, cursor ?? 0).map((a) => (cursor === null ? a * 0.3 : a)),
    [rec, trial, cursor],
  );

  const phase = !result || stage.index < SCORERS ? "none" : stage.index < SINKHORN ? "pre" : "post";
  const ranking = phase === "pre" ? result!.pre : phase === "post" ? result!.post : null;
  const row = ranking?.ranked[trial] ?? null;
  const prob = (id: number) => (ranking ? ranking.probs[trial * 102 + (id - 1001)] : undefined);
  const truth = result?.labels?.[trial];

  const activeStage = stages[Math.min(stage.index, stages.length - 1)];

  return (
    <div className="grid gap-5 lg:grid-cols-[320px_minmax(0,1fr)]">
      <aside className="card p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="card-title">Pipeline</h2>
          <button type="button" className="text-xs text-iris hover:underline" onClick={onSkip}>
            Skip to results →
          </button>
        </div>
        <PipelineStages stages={stages} index={stage.index} progress={stage.progress} />
        <p className="mt-3 text-[11px] leading-relaxed text-subtle">
          <span className="text-foam">in browser</span>: executed now on your file.{" "}
          <span className="text-gold">precomputed</span>: {info.name} ran offline in Python for {subject ?? "this subject"} (the fitted
          models are many GB and are not shipped); its fused 102 × 102 score matrix is loaded, and the browser runs the
          post-processing.
        </p>
      </aside>

      <div className="grid min-w-0 gap-5">
        <section className="card p-4" aria-live="polite">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h2 className="card-title">
              Analysing trial <span className="font-mono text-foam">{trial + 1}</span>
              <span className="text-subtle"> / {rec.nTrials}</span>
            </h2>
            <span className="text-xs text-subtle">
              {activeStage.title}
              {cursor !== null && <span className="ml-2 font-mono text-foam">t = {cursor.toFixed(2)} s</span>}
            </span>
          </div>
          <EegCanvas rec={rec} trial={trial} cursor={cursor} activity={activity} windows={stage.index >= 3 ? info.windows : []} height={300} />
        </section>

        <div className="grid gap-5 md:grid-cols-2">
          <section className="card flex flex-col p-4">
            <h2 className="card-title">Scalp heatmap</h2>
            <div className="flex flex-1 items-center justify-center pt-2">
              <ScalpHeatmap
                activity={activity}
                label="Signal strength (RMS) under the cursor, per electrode, interpolated between electrodes. Colours are relative to the other channels at this moment."
              />
            </div>
          </section>

          <section className="card p-4">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="card-title">Paired video</h2>
              <span className="chip bg-overlay text-subtle">
                {phase === "none" ? "pending" : phase === "pre" ? "fused scores" : "after bijection"}
              </span>
            </div>
            <div className="grid grid-cols-[minmax(0,1fr)_88px] items-start gap-3">
              <VideoCard
                id={row?.[0] ?? null}
                rank={row ? 1 : undefined}
                score={row ? prob(row[0]) : undefined}
                scoreLabel={phase === "post" ? "p" : "score"}
                isTruth={row !== null && truth === row[0]}
                autoPlay
                placeholder="The best-matching video appears once the scorers run."
              />
              <ol className="space-y-2" aria-label="Runner-up candidates">
                {(row?.slice(1, 5) ?? [null, null, null, null]).map((id, k) => (
                  <li key={k} className="animate-fade-in">
                    <VideoCard id={id} rank={id ? k + 2 : undefined} isTruth={id !== null && id === truth} compact placeholder="" />
                  </li>
                ))}
              </ol>
            </div>
            {truth !== undefined && row && (
              <p className="mt-3 text-xs text-subtle">
                Ground truth for this trial: <span className="font-mono text-foam">video {truth}</span>, ranked{" "}
                <span className="font-mono text-gold">#{row.indexOf(truth) + 1}</span>
              </p>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
