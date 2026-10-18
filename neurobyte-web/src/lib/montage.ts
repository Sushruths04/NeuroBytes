/**
 * Electrode montage of the challenge recordings.
 *
 * Names: supplied by the organisers on 18 Oct 2026, in file order (channel index
 * 0..16), all over parietal/occipital cortex. The project README records the list
 * and a check against the released training EEG (inter-channel correlation tracks
 * scalp distance, rho 0.89–0.94 per subject).
 *
 * Positions: the organisers gave names only, so these are the standard 10-10
 * positions for those names (EEGLAB-style polar coordinates: angle from the nose,
 * clockwise, negative = left; radius 0.5 ≈ the Fpz–Oz ring), converted to a top-down
 * x/y with the nose up. They are approximate, not digitised per person.
 */
export const CHANNELS = [
  "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "P1",
  "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2",
] as const;

export type Channel = (typeof CHANNELS)[number];

/** [angle in degrees from the nose (negative = left), radius] for each name. */
const POLAR: Record<Channel, [number, number]> = {
  Pz: [180, 0.256], POz: [180, 0.383], Oz: [180, 0.511],
  P1: [-157, 0.272], P2: [157, 0.272],
  P3: [-141, 0.333], P4: [141, 0.333],
  P5: [-137, 0.417], P6: [137, 0.417],
  P7: [-144, 0.511], P8: [144, 0.511],
  PO3: [-158, 0.42], PO4: [158, 0.42],
  PO7: [-153, 0.511], PO8: [153, 0.511],
  O1: [-162, 0.511], O2: [162, 0.511],
};

/** Top-down position per channel index: x right, y towards the nose, head edge at radius ~1. */
export const POSITIONS: [number, number][] = CHANNELS.map((name) => {
  const [deg, r] = POLAR[name];
  const a = (deg * Math.PI) / 180;
  const R = r * 1.8; // Oz ring at ~0.92 of the head outline
  return [R * Math.sin(a), R * Math.cos(a)];
});

/** "Pz" for index 0, etc.; falls back to the index for anything out of range. */
export const channelName = (i: number): string => CHANNELS[i] ?? `ch${i}`;
