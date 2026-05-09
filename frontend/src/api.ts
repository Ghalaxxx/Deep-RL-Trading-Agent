import type { DashboardData } from "./types";

const API_ROOT = "";

export async function fetchDashboard(ticker: string): Promise<DashboardData> {
  const response = await fetch(`${API_ROOT}/api/dashboard?ticker=${encodeURIComponent(ticker)}`);
  if (!response.ok) {
    throw new Error(`Dashboard API failed: ${response.status}`);
  }
  return response.json();
}
