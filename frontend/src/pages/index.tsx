import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ChangeEventHandler,
  type DragEventHandler,
} from "react";
import Head from "next/head";
import { useMutation } from "@tanstack/react-query";
import ResultDisplay, { DiseaseInfo } from "@/components/ResultDisplay";

const ACCEPTED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/jpg",
  "image/heic",
  "image/heif",
];

const MAX_FILE_SIZE_MB = 8;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

type PredictionResult = {
  prediction: string;
  confidence: number;
  info: DiseaseInfo;
};

type ApiErrorPayload = {
  error?: string;
};

const LoadingIndicator = () => (
  <div className="flex items-center gap-4 text-emerald-600">
    <span className="inline-flex h-12 w-12 items-center justify-center">
      <span className="h-12 w-12 rounded-full border-4 border-emerald-200 border-t-emerald-500 animate-spin" />
    </span>
    <span className="text-base font-semibold">Menganalisis gambar...</span>
  </div>
);

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const {
    mutateAsync: runPrediction,
    isPending,
    reset: resetPrediction,
  } = useMutation<PredictionResult, Error, File>({
    // HILANGKAN { signal } DI SINI
    mutationFn: async (selectedFile) => {
      const formData = new FormData();
      formData.append("file", selectedFile);

      let response: Response;

      try {
        response = await fetch(`${API_BASE_URL}/api/predict`, {
          method: "POST",
          body: formData,
          // HILANGKAN opsi signal: signal,
        });
      } catch {
        throw new Error("Koneksi ke server gagal. Periksa jaringan Anda.");
      }

      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        const message =
          (payload as ApiErrorPayload | null)?.error ??
          "Terjadi kesalahan saat memproses gambar.";
        throw new Error(message);
      }

      if (!payload) {
        throw new Error("Server tidak mengembalikan data prediksi.");
      }

      return payload as PredictionResult;
    },
    onMutate: () => {
      setErrorMessage(null);
    },
    onSuccess: (data) => {
      setResult(data);
    },
    onError: (error) => {
      setResult(null);
      setErrorMessage(error.message);
    },
  });

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const acceptedFormats = useMemo(
    () =>
      ACCEPTED_TYPES.map(
        (type) => type.split("/")[1]?.toUpperCase() ?? type,
      ).join(", "),
    [],
  );

  const validateAndStoreFile = useCallback((candidate: File) => {
    if (!ACCEPTED_TYPES.includes(candidate.type)) {
      setErrorMessage(
        "Format file tidak didukung. Gunakan JPEG, PNG, atau WebP.",
      );
      return;
    }
    if (candidate.size > MAX_FILE_SIZE_BYTES) {
      setErrorMessage(`Ukuran file melebihi ${MAX_FILE_SIZE_MB}MB.`);
      return;
    }
    setFile(candidate);
    setResult(null);
    setErrorMessage(null);
    resetPrediction();
  }, [resetPrediction]);

  const handleFileInputChange = useCallback<ChangeEventHandler<HTMLInputElement>>(
    (event) => {
      const files = event.target.files;
      if (!files || files.length === 0) {
        return;
      }
      validateAndStoreFile(files[0]);
    },
    [validateAndStoreFile],
  );

  const handleDrop = useCallback<DragEventHandler<HTMLLabelElement>>(
    (event) => {
      event.preventDefault();
      setIsDragging(false);
      if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
        validateAndStoreFile(event.dataTransfer.files[0]);
      }
    },
    [validateAndStoreFile],
  );

  const handleDragOver = useCallback<DragEventHandler<HTMLLabelElement>>(
    (event) => {
      event.preventDefault();
      setIsDragging(true);
    },
    [],
  );

  const handleDragLeave = useCallback<DragEventHandler<HTMLLabelElement>>(
    () => {
      setIsDragging(false);
    },
    [],
  );

  const handleSubmit = useCallback(async () => {
    if (!file) {
      setErrorMessage("Silakan pilih gambar daun terlebih dahulu.");
      return;
    }
    await runPrediction(file).catch(() => undefined);
  }, [file, runPrediction]);

  const resetSelection = useCallback(() => {
    setFile(null);
    setPreviewUrl("");
    setResult(null);
    setErrorMessage(null);
    resetPrediction();
  }, [resetPrediction]);

  return (
    <>
      <Head>
        <title>Detektor Penyakit Daun</title>
        <meta
          name="description"
          content="Analisis cepat penyakit daun anggur, apel, jagung, kentang, dan tomat berbasis machine learning."
        />
      </Head>

      <div className="min-h-screen flex flex-col relative">
        {/* Background decorative elements */}
        <div className="fixed inset-0 pointer-events-none z-0">
          {/* Floating orbs */}
          <div className="absolute top-20 left-10 w-32 h-32 bg-gradient-to-br from-emerald-400/20 to-transparent rounded-full blur-xl animate-pulse"></div>
          <div className="absolute top-40 right-20 w-24 h-24 bg-gradient-to-br from-lime-400/20 to-transparent rounded-full blur-lg animate-pulse delay-1000"></div>
          <div className="absolute bottom-40 left-1/4 w-40 h-40 bg-gradient-to-br from-green-400/15 to-transparent rounded-full blur-2xl animate-pulse delay-2000"></div>
          <div className="absolute bottom-20 right-1/3 w-28 h-28 bg-gradient-to-br from-emerald-300/20 to-transparent rounded-full blur-xl animate-pulse delay-500"></div>

          {/* Leaf-like shapes */}
          <div className="absolute top-1/3 left-1/4 w-16 h-16 bg-gradient-to-br from-green-300/10 to-transparent rounded-full blur-sm transform rotate-45 animate-pulse delay-700"></div>
          <div className="absolute top-2/3 right-1/4 w-20 h-20 bg-gradient-to-br from-emerald-300/10 to-transparent rounded-full blur-sm transform -rotate-12 animate-pulse delay-1500"></div>
          <div className="absolute top-1/2 left-1/2 w-12 h-12 bg-gradient-to-br from-lime-300/15 to-transparent rounded-full blur-sm transform rotate-30 animate-pulse delay-3000"></div>

          {/* Subtle grid pattern */}
          <div className="absolute inset-0 opacity-5">
            <div
              className="w-full h-full"
              style={{
                backgroundImage: `
                linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px),
                linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px)
              `,
                backgroundSize: "50px 50px",
              }}
            ></div>
          </div>
        </div>

        <main className="flex-1 py-16 sm:py-20 relative z-10">
          <div className="mx-auto flex w-full max-w-6xl flex-col gap-12 px-4 sm:px-8">
            {/* Header Section */}
            <section className="flex flex-col items-center text-center floating-elements">
              <span className="inline-flex w-fit items-center gap-2 rounded-full bg-emerald-100 px-4 py-1 text-sm font-semibold text-emerald-700">
                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                Vision AI untuk pertanian presisi
              </span>
              <h1 className="mt-4 text-4xl font-semibold leading-tight text-emerald-900 sm:text-5xl">
                Detektor Penyakit Daun
              </h1>
              <p className="mt-4 max-w-3xl text-lg text-slate-700">
                Unggah foto daun tanaman Anda dan dapatkan diagnosis cepat
                lengkap dengan langkah pengobatan, perawatan, serta
                pencegahan. Model ini dilatih untuk mendeteksi penyakit pada
                anggur, apel, jagung, kentang, dan tomat.
              </p>
            </section>

            {/* Features Section */}
            <section className="flex justify-center leaf-pattern">
              <ul className="grid gap-3 text-slate-600 sm:grid-cols-2">
                <li className="flex items-start gap-3 rounded-2xl bg-white/90 p-4 shadow-soft border border-emerald-100">
                  <span className="mt-1 h-2 w-2 rounded-full bg-emerald-600" />
                  <div>
                    <p className="font-semibold text-emerald-900">
                      Analisis instan
                    </p>
                    <p className="text-sm text-slate-700">
                      Hasil prediksi beserta keyakinan model tampil dalam
                      hitungan detik.
                    </p>
                  </div>
                </li>
                <li className="flex items-start gap-3 rounded-2xl bg-white/90 p-4 shadow-soft border border-emerald-100">
                  <span className="mt-1 h-2 w-2 rounded-full bg-emerald-600" />
                  <div>
                    <p className="font-semibold text-emerald-900">
                      Panduan agronomi
                    </p>
                    <p className="text-sm text-slate-700">
                      Dapatkan deskripsi, pengobatan, perawatan, dan
                      pencegahan yang relevan.
                    </p>
                  </div>
                </li>
              </ul>
            </section>

            {/* Upload Card Section - Landscape Layout */}
            <section className="flex justify-center floating-elements">
              <div className="glass-panel w-full max-w-6xl rounded-3xl border border-emerald-100 p-10 shadow-soft">
                <div className="flex flex-col gap-8 lg:flex-row lg:items-center lg:gap-12">
                  {/* Upload Area - Left Side */}
                  <div className="flex-1">
                    <label
                      htmlFor="leaf-upload"
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                      className={`flex cursor-pointer flex-col items-center justify-center gap-6 rounded-2xl border-2 border-dashed p-10 text-center transition-all duration-200 ${
                        isDragging
                          ? "border-emerald-500 bg-emerald-50"
                          : "border-emerald-200 bg-white/60 hover:border-emerald-400"
                      }`}
                    >
                      <span className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald-100 text-emerald-600">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          viewBox="0 0 24 24"
                          fill="currentColor"
                          className="h-8 w-8"
                          aria-hidden
                        >
                          <path d="M12 3a5 5 0 0 1 5 5v1h.75A2.25 2.25 0 0 1 20 11.25v5.5A2.25 2.25 0 0 1 17.75 19H6.25A2.25 2.25 0 0 1 4 16.75v-5.5A2.25 2.25 0 0 1 6.25 9H7V8a5 5 0 0 1 5-5Zm0 2a3 3 0 0 0-3 3v1h6V8a3 3 0 0 0-3-3Zm-.75 5.5a.75.75 0 0 1 1.5 0v3.19l1.22-1.22a.75.75 0 1 1 1.06 1.06l-2.5 2.5a.75.75 0 0 1-1.06 0l-2.5-2.5a.75.75 0 1 1 1.06-1.06l1.22 1.22Z" />
                        </svg>
                      </span>
                      <div className="flex flex-col gap-2">
                        <p className="text-lg font-semibold text-emerald-900">
                          Tarik dan letakkan gambar daun
                        </p>
                        <p className="text-base text-slate-600">
                          atau klik untuk memilih dari perangkat Anda
                        </p>
                        <p className="text-sm text-slate-500">
                          Format: {acceptedFormats} | Maks {MAX_FILE_SIZE_MB}MB
                        </p>
                      </div>
                      <input
                        id="leaf-upload"
                        name="leaf-upload"
                        type="file"
                        accept={ACCEPTED_TYPES.join(",")}
                        onChange={handleFileInputChange}
                        className="hidden"
                      />
                    </label>
                  </div>

                  {/* Preview and Actions - Right Side */}
                  <div className="flex flex-col gap-6 lg:w-96">
                    {previewUrl && (
                      <div className="overflow-hidden rounded-2xl border border-emerald-100 shadow-soft">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={previewUrl}
                          alt="Pratinjau daun yang dipilih"
                          className="aspect-square w-full object-cover"
                        />
                      </div>
                    )}

                    <div className="flex flex-col gap-4">
                      <div className="flex gap-3">
                        <button
                          type="button"
                          onClick={handleSubmit}
                          disabled={!file || isPending}
                          className="flex-1 rounded-full bg-gradient-to-r from-emerald-600 to-emerald-500 px-6 py-4 text-base font-semibold text-white shadow-lg shadow-emerald-500/30 transition-all hover:from-emerald-700 hover:to-emerald-600 hover:shadow-emerald-600/40 hover:scale-105 disabled:cursor-not-allowed disabled:from-emerald-300 disabled:to-emerald-200 disabled:text-emerald-500 disabled:shadow-none disabled:scale-100"
                        >
                          Analisis Gambar
                        </button>
                        {file && (
                          <button
                            type="button"
                            onClick={resetSelection}
                            className="rounded-full border-2 border-emerald-500 bg-white/90 px-6 py-4 text-base font-semibold text-emerald-600 transition-all hover:bg-emerald-500 hover:text-white hover:shadow-lg hover:shadow-emerald-500/30 hover:scale-105"
                          >
                            Ganti
                          </button>
                        )}
                      </div>
                      {isPending && <LoadingIndicator />}
                    </div>
                  </div>
                </div>

                {errorMessage && (
                  <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                    {errorMessage}
                  </div>
                )}
              </div>
            </section>

            {result && (
              <ResultDisplay
                prediction={result.prediction}
                confidence={result.confidence}
                info={result.info}
              />
            )}
          </div>
        </main>

        <footer className="border-t border-emerald-100 bg-white/70 py-6">
          <div className="mx-auto flex w-full max-w-6xl flex-col gap-2 px-4 text-sm text-slate-500 sm:flex-row sm:items-center sm:justify-between">
            <p>
              &copy; {new Date().getFullYear()} Project Daun. Semua hak
              dilindungi.
            </p>
            <p>
              Terhubung ke API:{" "}
              <span className="font-semibold text-emerald-700">
                {API_BASE_URL}
              </span>
            </p>
          </div>
        </footer>
      </div>
    </>
  );
}
