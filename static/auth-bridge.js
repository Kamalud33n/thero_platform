/**
 * auth-bridge.js — MedNova Care bridge token handling (integration plan step 7).
 *
 * thero's API routes are JWT-protected (auth.get_current_therapist). This
 * script is what actually gets that token into every /api/* request, which
 * was missing before — that's why requests were coming back 401.
 *
 * Token sources, in priority order:
 *   1. postMessage from the parent window — used once thero is embedded in
 *      an <iframe> inside MedNova Care: {type: 'mednova_token', token: '...'}
 *   2. ?token=... on the URL — used the first time MedNova Care loads the
 *      iframe/page. Stripped from the visible URL immediately after read.
 *   3. sessionStorage fallback — ONLY so the token survives normal link
 *      clicks/page navigation while testing thero directly in a browser
 *      tab (no iframe parent). Per the integration plan this should be
 *      memory-only once real iframe embedding is live; this fallback exists
 *      purely for local/dev testing and can be deleted at that point.
 *
 * Load this BEFORE app.js / any inline <script> that calls fetch().
 */
(function () {
    const STORAGE_KEY = 'thero_bridge_token';
    let token = null;

    function setToken(t) {
        if (!t) return;
        token = t;
        try { sessionStorage.setItem(STORAGE_KEY, t); } catch (e) { /* private mode etc. */ }
    }

    // 1. iframe parent handoff
    window.addEventListener('message', function (event) {
        if (event.data && event.data.type === 'mednova_token' && typeof event.data.token === 'string') {
            setToken(event.data.token);
        }
    });

    // 2. URL param (first load) — tolerate stray whitespace in the key,
    // since a malformed upstream URL (e.g. "? token =...") would otherwise
    // silently fail with URLSearchParams.get('token') === null, and the
    // request would 401 with no visible error anywhere in the chain.
    const params = new URLSearchParams(window.location.search);
    let urlToken = params.get('token');
    if (!urlToken) {
        for (const [key, value] of params.entries()) {
            if (key.trim() === 'token') {
                urlToken = value;
                console.warn('bridge token found under malformed key:', JSON.stringify(key));
                break;
            }
        }
    }

    if (urlToken) {
        setToken(urlToken);
        const cleanUrl = window.location.pathname + window.location.hash;
        window.history.replaceState({}, document.title, cleanUrl);
    } else {
        // 3. dev/testing fallback
        try {
            const stored = sessionStorage.getItem(STORAGE_KEY);
            if (stored) token = stored;
        } catch (e) { /* private mode etc. */ }
    }

    window.getBridgeToken = function () { return token; };
    window.setBridgeToken = setToken; // exposed for a manual dev-console login helper if needed

    // Patch fetch so every same-origin /api/ and /integration/ call
    // automatically carries the bridge token, without editing every call
    // site individually.
    const originalFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
        const url = typeof input === 'string' ? input : (input && input.url) || '';
        const isProtected = url.startsWith('/api/') || url.startsWith('/integration/');
        if (isProtected && token) {
            init = init || {};
            const headers = new Headers(init.headers || {});
            if (!headers.has('Authorization')) {
                headers.set('Authorization', 'Bearer ' + token);
            }
            init = Object.assign({}, init, { headers });
        }
        return originalFetch(input, init);
    };
})();