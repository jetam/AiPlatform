export const BASE_URL = "http://127.0.0.1:5000";

export async function apiFetch(path, options = {}) {
//    credentials: "include" is required for the browser to send/accept the
//    session_id cookie across origins (frontend :3000 -> backend :5000) -
//    without it every request looks like a brand new session to the backend
    const response = await fetch(`${BASE_URL}${path}`, { credentials: "include", ...options });
    if (!response.ok) {
        let detail = response.statusText;
        try {
            const body = await response.clone().json();
            detail = body.detail || body.status || detail;
        } catch (_) {
            // response body wasn't json or was empty - fall back to statusText
        }
        throw new Error(`API error ${response.status}: ${detail}`);
    }
    return response;
}
