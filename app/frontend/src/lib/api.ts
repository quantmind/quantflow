function apiOrigin(): string {
  return (
    document
      .querySelector('meta[name="quantflow-api-origin"]')
      ?.getAttribute("content") || ""
  );
}

export async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${apiOrigin()}${path}`);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${path}`);
  }
  return response.json() as Promise<T>;
}
