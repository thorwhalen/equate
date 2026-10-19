/** The app shell: the setup screen, or the review screen for `?session=<id>`. */
import { useEffect } from 'react';
import { CostSummary } from './components/CostSummary';
import { LabelWidget } from './components/LabelWidget';
import { PairDetail } from './components/PairDetail';
import { ReviewTable } from './components/ReviewTable';
import { SetupForm } from './components/SetupForm';
import { Button } from './components/ui/button';
import { navigate, useRoute } from './route';
import { useReview } from './state/instance';

function ReviewScreen({ sessionId }: { sessionId: string }) {
  const session = useReview((s) => s.session);
  const status = useReview((s) => s.status);
  const load = useReview((s) => s.load);

  useEffect(() => {
    load(sessionId);
  }, [sessionId, load]);

  if (status === 'loading' || (!session && status !== 'error')) {
    return (
      <div aria-busy="true" className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <div className="h-96 animate-pulse rounded-xl bg-muted" />
        <div className="h-96 animate-pulse rounded-xl bg-muted" />
      </div>
    );
  }
  if (!session) {
    return (
      <div className="flex flex-col items-start gap-3">
        <p>This review session could not be opened. It may have expired with the server.</p>
        <Button onClick={() => navigate({ session: null })}>Start a new session</Button>
      </div>
    );
  }
  const confirmed = session.review.filter((r) => r.status === 'confirmed').length;
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm">
          {session.n_left} × {session.n_right} items · {session.assignment.length} matched ·{' '}
          {confirmed} confirmed
        </p>
        <CostSummary accounting={session.accounting} />
      </div>
      <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <div className="flex min-w-0 flex-col gap-4">
          <ReviewTable />
          <LabelWidget />
        </div>
        <PairDetail />
      </div>
    </div>
  );
}

export function App() {
  const route = useRoute();
  const error = useReview((s) => s.error);
  const dismiss = useReview((s) => s.dismissError);

  useEffect(() => {
    document.title = route.session ? 'Review · equate' : 'New session · equate';
  }, [route.session]);

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-6">
      <header className="flex items-center justify-between gap-4">
        <h1 className="text-lg font-semibold">equate review</h1>
        {route.session && (
          <Button variant="outline" size="sm" onClick={() => navigate({ session: null })}>
            New session
          </Button>
        )}
      </header>
      {error && (
        <div role="alert" className="flex items-start justify-between gap-4 rounded-md border border-destructive p-3 text-sm text-destructive">
          <span>{error}</span>
          <Button variant="ghost" size="sm" onClick={dismiss}>
            Dismiss
          </Button>
        </div>
      )}
      <main>{route.session ? <ReviewScreen sessionId={route.session} /> : <SetupForm />}</main>
    </div>
  );
}
