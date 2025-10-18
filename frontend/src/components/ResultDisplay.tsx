import { useMemo, useState } from "react";

type DiseaseInfo = {
  nama?: string;
  deskripsi?: string;
  pengobatan?: string;
  perawatan?: string;
  pencegahan?: string;
};

type ResultDisplayProps = {
  prediction: string;
  confidence: number;
  info: DiseaseInfo;
};

const TAB_ITEMS: Array<{ key: keyof DiseaseInfo; label: string }> = [
  { key: "deskripsi", label: "Deskripsi" },
  { key: "pengobatan", label: "Pengobatan" },
  { key: "perawatan", label: "Perawatan" },
  { key: "pencegahan", label: "Pencegahan" },
];

const formatPredictionLabel = (raw: string, fallbackName?: string) => {
  if (fallbackName && fallbackName.trim().length > 0) {
    return fallbackName;
  }
  return raw.replace(/_/g, " ");
};

export default function ResultDisplay({
  prediction,
  confidence,
  info,
}: ResultDisplayProps) {
  const [activeTab, setActiveTab] = useState<keyof DiseaseInfo>("deskripsi");

  const safeConfidence = useMemo(() => {
    if (Number.isNaN(confidence)) {
      return 0;
    }
    return Math.min(100, Math.max(0, Number(confidence)));
  }, [confidence]);

  const description = useMemo(() => {
    const selected = info?.[activeTab];
    return selected && selected.trim().length > 0
      ? selected
      : "Informasi belum tersedia untuk bagian ini.";
  }, [info, activeTab]);

  const displayName = useMemo(
    () => formatPredictionLabel(prediction, info?.nama),
    [prediction, info?.nama],
  );

  return (
    <section className="glass-panel shadow-soft rounded-3xl p-8 md:p-10 w-full flex flex-col gap-8 bg-white/80">
      <header className="flex flex-col gap-2">
        <p className="text-sm uppercase tracking-[0.3em] text-emerald-500">Hasil Analisis</p>
        <h2 className="text-3xl md:text-4xl font-semibold text-emerald-900">{displayName}</h2>
        <p className="text-slate-600">
          Model mendeteksi kondisi di atas pada citra yang Anda unggah. Gunakan informasi berikut sebagai panduan awal
          dan kombinasikan dengan observasi lapangan.
        </p>
      </header>

      <div className="flex flex-col gap-3 w-full md:w-auto">
        <span className="text-sm font-medium text-emerald-700">Keyakinan Model</span>
        <div className="w-full md:w-80 h-3 rounded-full bg-emerald-100 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-emerald-400 via-green-400 to-lime-300 transition-all duration-500"
            style={{ width: `${safeConfidence}%` }}
            aria-hidden
          />
        </div>
        <div className="flex justify-between text-sm text-slate-700 w-full md:w-80">
          <span>0%</span>
          <span className="font-semibold text-emerald-600">{safeConfidence.toFixed(1)}%</span>
        </div>
      </div>

      <nav className="grid grid-cols-2 sm:flex sm:flex-wrap gap-3" aria-label="Informasi penyakit">
        {TAB_ITEMS.map((tab) => {
          const isActive = tab.key === activeTab;
          return (
            <button
              key={tab.key as string}
              type="button"
              onClick={() => setActiveTab(tab.key)}
              className={`rounded-full px-5 py-2 text-sm font-medium transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:ring-offset-2 ${
                isActive
                  ? "bg-gradient-to-r from-emerald-600 to-emerald-500 text-white shadow-lg shadow-emerald-500/30 scale-105"
                  : "bg-white/90 text-emerald-700 border border-emerald-200 hover:bg-gradient-to-r hover:from-emerald-50 hover:to-emerald-100 hover:border-emerald-300 hover:text-emerald-800 hover:shadow-md hover:shadow-emerald-200/50 hover:scale-105 hover:-translate-y-0.5"
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </nav>

      <article className="rounded-2xl border border-emerald-100 bg-white/90 p-6 text-slate-700 leading-relaxed">
        {description}
      </article>
    </section>
  );
}

export type { DiseaseInfo };
