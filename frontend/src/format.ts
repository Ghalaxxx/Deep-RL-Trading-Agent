export function formatMetric(value: number | null, format: string): string {
  if (value === null || Number.isNaN(value)) return "N/A";
  if (format === "percent") return `${(value * 100).toFixed(1)}%`;
  if (format === "integer") return value.toFixed(0);
  if (format === "days") return `${value.toFixed(1)}d`;
  return value.toFixed(2);
}

export function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(value);
}

export function formatCompact(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1
  }).format(value);
}

export function labelForStrategy(strategy: string): string {
  return strategy
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
