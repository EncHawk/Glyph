'use client';

import { MarkdownBlock } from '@/app/components/canvas/markdown-block';
import type { Camera, CanvasCard } from '@/app/components/canvas/types';

export function CanvasOverlay({
  cards,
  camera,
  onStartDrag,
  onRegenerate,
}: {
  cards: CanvasCard[];
  camera: Camera;
  onStartDrag: (event: React.PointerEvent<HTMLButtonElement>, cardId: string) => void;
  onRegenerate: (cardId: string) => void;
}) {
  return (
    <div className="canvas-card-layer">
      {cards.map((card) => {
        const screenX = card.x * camera.zoom + camera.x;
        const screenY = card.y * camera.zoom + camera.y;

        return (
          <article
            key={card.id}
            className={`canvas-card ${card.kind}`}
            style={{
              width: `${card.width}px`,
              transform: `translate(${screenX}px, ${screenY}px) scale(${camera.zoom})`,
            }}
          >
            <button
              type="button"
              className="card-handle"
              onPointerDown={(event) => onStartDrag(event, card.id)}
            >
              <div>
                <p className="card-mode">{card.mode === 'video' ? 'Video' : 'Text'}</p>
                <h2>{card.prompt}</h2>
              </div>
              <span className="card-status">{card.statusLabel}</span>
            </button>

            <div className="card-body">
              <div className="card-meta">
                <span>{card.route || card.mode}</span>
                <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                  {card.warnings.length > 0 ? <span>{card.warnings.length} warning(s)</span> : null}
                  {card.kind !== 'pending' && (
                    <button 
                      type="button" 
                      onClick={() => onRegenerate(card.id)}
                      title="Regenerate"
                      style={{
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        padding: 0,
                        display: 'flex',
                        color: 'var(--text-secondary)'
                      }}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 2v6h-6"></path>
                        <path d="M3 12a9 9 0 0 1 15-6.7L21 8"></path>
                        <path d="M3 22v-6h6"></path>
                        <path d="M21 12a9 9 0 0 1-15 6.7L3 16"></path>
                      </svg>
                    </button>
                  )}
                </div>
              </div>

              {card.kind === 'pending' ? (
                <div className="pending-block">
                  <span className="spinner" />
                  <p>Generating response...</p>
                </div>
              ) : null}

              {card.kind === 'error' ? (
                <section className="card-section error-section">
                  <h3>Error</h3>
                  <MarkdownBlock content={card.error} />
                </section>
              ) : null}

              {card.content ? (
                <section className="card-section">
                  <h3>{card.mode === 'video' ? 'Summary' : 'Answer'}</h3>
                  <MarkdownBlock content={card.content} />
                </section>
              ) : null}

              {card.kind === 'video' && card.response ? (
                <section className="card-section video-section">
                  <h3>Video</h3>
                  <video controls playsInline preload="metadata" src={card.response} />
                  <a href={card.response} target="_blank" rel="noreferrer">
                    Open MP4
                  </a>
                </section>
              ) : null}

              {card.research ? (
                <details className="card-section research-section">
                  <summary>Sources</summary>
                  <MarkdownBlock content={card.research} />
                </details>
              ) : null}

              {card.warnings.length > 0 ? (
                <section className="card-section warning-section">
                  <h3>Warnings</h3>
                  <ul>
                    {card.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </section>
              ) : null}
            </div>
          </article>
        );
      })}
    </div>
  );
}
