/**
 * The one module that owns navigation (tw-frontend-ux principle 2). The URL is the state:
 * `?session=<id>` is the review screen, no parameter is the setup screen. Opening a
 * session is a new history entry, so Back returns to setup and never leaves the app.
 */
import { useSyncExternalStore } from 'react';

export interface Route {
  session: string | null;
}

const EVENT = 'equate:navigate';

export function parseRoute(search: string): Route {
  return { session: new URLSearchParams(search).get('session') };
}

export function formatRoute(route: Route): string {
  return route.session ? `?session=${encodeURIComponent(route.session)}` : location.pathname;
}

export function navigate(route: Route, { replace = false } = {}) {
  const url = formatRoute(route);
  if (url === `${location.search}` || (url === location.pathname && !location.search)) return;
  if (replace) history.replaceState(null, '', url);
  else history.pushState(null, '', url);
  window.dispatchEvent(new Event(EVENT));
  if (!replace) window.scrollTo(0, 0);
}

function subscribe(cb: () => void) {
  window.addEventListener('popstate', cb);
  window.addEventListener(EVENT, cb);
  return () => {
    window.removeEventListener('popstate', cb);
    window.removeEventListener(EVENT, cb);
  };
}

export function useRoute(): Route {
  const search = useSyncExternalStore(subscribe, () => location.search);
  return parseRoute(search);
}
