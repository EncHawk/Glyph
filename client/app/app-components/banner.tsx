'use client';

import { useEffect, useState } from 'react';

const DISMISS_KEY = 'glyph-beta-banner-dismissed';

export function Banner() {
  const [isDismissed, setIsDismissed] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setIsDismissed(window.localStorage.getItem(DISMISS_KEY) === 'true');
    setMounted(true);
  }, []);

  const handleDismiss = () => {
    window.localStorage.setItem(DISMISS_KEY, 'true');
    setIsDismissed(true);
  };

  if (!mounted || isDismissed) {
    return null;
  }

  return (
    <section className="beta-banner" aria-label="Beta notice">
      <div className="beta-banner__copy">
        <span className="beta-banner__eyebrow">Glyph AI Beta</span>
        <p>
          Expect rapid iteration, rough edges, and the occasional breaking change. <br/>
          <span className='text-md text-shadow-2xs text-sky-200'>I am grateful for you continued support, awaiting your review :)</span>
        </p>
      </div>
      <button
        type="button"
        className="beta-banner__close flex items-center justify-center p-0"
        aria-label="Dismiss beta notice"
        onClick={handleDismiss}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </section>
  );
}
