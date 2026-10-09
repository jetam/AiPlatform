export const BASE_URL = "http://127.0.0.1:5000";

export async function apiFetch(path, options = {}) {
    const response = await fetch(`${BASE_URL}${path}`, options);
    if (!response.ok) {
        let detail = response.statusText;
        try {
            const body = await response.clone().json();
            detail = body.detail || body.status || detail;
        } catch (_) {
            // response body wasn't json (or was empty) - fall back to statusText
        }
        throw new Error(`API error ${response.status}: ${detail}`);
    }
    return response;
}
