'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import MarkdownIt from 'markdown-it';

type Mode = 'text' | 'video';
type CardKind = 'pending' | 'text' | 'video' | 'error';

type ApiResponse = {
  ok: boolean;
  route?: string | null;
  response?: string | null;
  content?: unknown;
  research?: unknown;
  error?: string | null;
  warnings?: string[];
};

type CanvasCard = {
  id: string;
  kind: CardKind;
  mode: Mode;
  prompt: string;
  x: number;
  y: number;
  width: number;
  content: string;
  research: string;
  response: string;
  route: string;
  error: string;
  warnings: string[];
  statusLabel: string;
};

type DragState = {
  cardId: string;
  offsetX: number;
  offsetY: number;
};

type Camera = {
  x: number;
  y: number;
  zoom: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://127.0.0.1:8080';
const CARD_WIDTH = 380;
const CARD_STACK_X = 440;
const CARD_STACK_Y = 340;
const CARD_HEIGHT_ESTIMATE = 440;
const MIN_ZOOM = 0.15;
const MAX_ZOOM = 3;
const ZOOM_SENSITIVITY = 0.0015;

const markdown = new MarkdownIt({ html: false, linkify: true, breaks: true });

/* ── coordinate helpers ─────────────────────────────────────────────── */

function screenToCanvas(sx: number, sy: number, cam: Camera) {
  return {
    x: (sx - cam.x) / cam.zoom,
    y: (sy - cam.y) / cam.zoom,
  };
}

/* ── data helpers ───────────────────────────────────────────────────── */

function normalizeText(value: unknown): string {
  if (typeof value === 'string') return value.trim();
  if (value == null) return '';
  if (Array.isArray(value))
    return value.map(normalizeText).filter(Boolean).join('\n');
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2).trim();
    } catch {
      return String(value).trim();
    }
  }
  return String(value).trim();
}

function makePendingCard(
  id: string,
  mode: Mode,
  prompt: string,
  x: number,
  y: number,
): CanvasCard {
  return {
    id,
    kind: 'pending',
    mode,
    prompt,
    x,
    y,
    width: CARD_WIDTH,
    content: '',
    research: '',
    response: '',
    route: mode,
    error: '',
    warnings: [],
    statusLabel: 'Running',
  };
}

function toResolvedCard(card: CanvasCard, data: ApiResponse): CanvasCard {
  const content = normalizeText(data.content ?? data.response);
  const research = normalizeText(data.research);
  const response = normalizeText(data.response);
  const error = normalizeText(data.error);
  const warnings = (data.warnings ?? []).map((w) => w.trim()).filter(Boolean);

  if (!data.ok) {
    return {
      ...card,
      kind: 'error',
      content,
      research,
      response,
      route: data.route ?? card.route,
      error: error || 'Request failed.',
      warnings,
      statusLabel: 'Failed',
    };
  }

  return {
    ...card,
    kind: card.mode === 'video' ? 'video' : 'text',
    content,
    research,
    response,
    route: data.route ?? card.route,
    error: '',
    warnings,
    statusLabel: card.mode === 'video' ? 'Rendered' : 'Answered',
  };
}

/* ── markdown renderer ─────────────────────────────────────────────── */

function MarkdownBlock({ content }: { content: string }) {
  return (
    <div
      className="markdown-block"
      dangerouslySetInnerHTML={{ __html: markdown.render(content) }}
    />
  );
}

/* ── SVG Icons ──────────────────────────────────────────────────────── */

function ChevronUpIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="18 15 12 9 6 15" />
    </svg>
  );
}

function ChevronDownIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function CrosshairIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="22" y1="12" x2="18" y2="12" />
      <line x1="6" y1="12" x2="2" y2="12" />
      <line x1="12" y1="6" x2="12" y2="2" />
      <line x1="12" y1="22" x2="12" y2="18" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" />
      <line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" />
      <line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

/* ── canvas card overlay ────────────────────────────────────────────── */

function CanvasOverlay({
  cards,
  camera,
  onStartDrag,
}: {
  cards: CanvasCard[];
  camera: Camera;
  onStartDrag: (event: React.PointerEvent<HTMLButtonElement>, cardId: string) => void;
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
              onPointerDown={(e) => onStartDrag(e, card.id)}
            >
              <div>
                <p className="card-mode">
                  {card.mode === 'video' ? 'Video' : 'Text'}
                </p>
                <h2>{card.prompt}</h2>
              </div>
              <span className="card-status">{card.statusLabel}</span>
            </button>

            <div className="card-body">
              <div className="card-meta">
                <span>{card.route || card.mode}</span>
                {card.warnings.length > 0 && (
                  <span>{card.warnings.length} warning(s)</span>
                )}
              </div>

              {card.kind === 'pending' && (
                <div className="pending-block">
                  <span className="spinner" />
                  <p>Generating response…</p>
                </div>
              )}

              {card.kind === 'error' && (
                <section className="card-section error-section">
                  <h3>Error</h3>
                  <MarkdownBlock content={card.error} />
                </section>
              )}

              {card.content && (
                <section className="card-section">
                  <h3>{card.mode === 'video' ? 'Summary' : 'Answer'}</h3>
                  <MarkdownBlock content={card.content} />
                </section>
              )}

              {card.kind === 'video' && card.response && (
                <section className="card-section video-section">
                  <h3>Video</h3>
                  <video controls playsInline preload="metadata" src={card.response} />
                  <a href={card.response} target="_blank" rel="noreferrer">
                    Open MP4
                  </a>
                </section>
              )}

              {card.research && (
                <details className="card-section research-section">
                  <summary>Sources</summary>
                  <MarkdownBlock content={card.research} />
                </details>
              )}

              {card.warnings.length > 0 && (
                <section className="card-section warning-section">
                  <h3>Warnings</h3>
                  <ul>
                    {card.warnings.map((w) => (
                      <li key={w}>{w}</li>
                    ))}
                  </ul>
                </section>
              )}
            </div>
          </article>
        );
      })}
    </div>
  );
}

/* ── main page ──────────────────────────────────────────────────────── */

export default function Home() {
  const [mode, setMode] = useState<Mode>('text');
  const [prompt, setPrompt] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [cards, setCards] = useState<CanvasCard[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [dockExpanded, setDockExpanded] = useState(false);
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');

  const [camera, setCamera] = useState<Camera>({ x: 0, y: 0, zoom: 1 });

  const canvasRef = useRef<HTMLDivElement>(null);
  const compactTextareaRef = useRef<HTMLTextAreaElement>(null);
  const expandedTextareaRef = useRef<HTMLTextAreaElement>(null);
  const placementIndexRef = useRef(0);
  const dragStateRef = useRef<DragState | null>(null);
  const isPanningRef = useRef(false);
  const panStartRef = useRef({ x: 0, y: 0, camX: 0, camY: 0 });
  const spaceHeldRef = useRef(false);

  // Generate session ID
  useEffect(() => {
    setSessionId(crypto.randomUUID());
  }, []);

  /* ── keyboard: space held for pan ─────────────────────────────────── */
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat) {
        // Don't interfere if typing in the dock
        const tag = (e.target as HTMLElement)?.tagName;
        if (tag === 'TEXTAREA' || tag === 'INPUT') return;
        e.preventDefault();
        spaceHeldRef.current = true;
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        spaceHeldRef.current = false;
      }
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  /* ── wheel: scroll=pan, ctrl/meta+scroll=zoom ─────────────────────── */
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();

      if (e.ctrlKey || e.metaKey) {
        // Zoom
        setCamera((cam) => {
          const zoomFactor = 1 - e.deltaY * ZOOM_SENSITIVITY;
          const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, cam.zoom * zoomFactor));
          // Zoom towards cursor
          const ratio = newZoom / cam.zoom;
          return {
            x: e.clientX - (e.clientX - cam.x) * ratio,
            y: e.clientY - (e.clientY - cam.y) * ratio,
            zoom: newZoom,
          };
        });
      } else {
        // Pan
        setCamera((cam) => ({
          ...cam,
          x: cam.x - e.deltaX,
          y: cam.y - e.deltaY,
        }));
      }
    };

    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  /* ── pointer: pan (middle-click or space+left) & card drag ────────── */
  useEffect(() => {
    const onPointerMove = (e: PointerEvent) => {
      // Card drag
      const drag = dragStateRef.current;
      if (drag) {
        const canvasPoint = screenToCanvas(e.clientX, e.clientY, camera);
        setCards((prev) =>
          prev.map((c) =>
            c.id === drag.cardId
              ? { ...c, x: canvasPoint.x - drag.offsetX, y: canvasPoint.y - drag.offsetY }
              : c,
          ),
        );
        return;
      }

      // Pan
      if (isPanningRef.current) {
        const dx = e.clientX - panStartRef.current.x;
        const dy = e.clientY - panStartRef.current.y;
        setCamera((cam) => ({
          ...cam,
          x: panStartRef.current.camX + dx,
          y: panStartRef.current.camY + dy,
        }));
      }
    };

    const onPointerUp = () => {
      dragStateRef.current = null;
      isPanningRef.current = false;
    };

    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
    };
  }, [camera]);

  const handleCanvasPointerDown = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      // Middle-click or space+left-click = pan
      if (e.button === 1 || (e.button === 0 && spaceHeldRef.current)) {
        e.preventDefault();
        isPanningRef.current = true;
        panStartRef.current = {
          x: e.clientX,
          y: e.clientY,
          camX: camera.x,
          camY: camera.y,
        };
      }
    },
    [camera],
  );

  const beginCardDrag = useCallback(
    (e: React.PointerEvent<HTMLButtonElement>, cardId: string) => {
      e.preventDefault();
      e.stopPropagation();

      const targetCard = cards.find((c) => c.id === cardId);
      if (!targetCard) return;

      const canvasPoint = screenToCanvas(e.clientX, e.clientY, camera);
      dragStateRef.current = {
        cardId,
        offsetX: canvasPoint.x - targetCard.x,
        offsetY: canvasPoint.y - targetCard.y,
      };
    },
    [cards, camera],
  );

  /* ── actions ──────────────────────────────────────────────────────── */

  const resetSession = useCallback(() => {
    setSessionId(crypto.randomUUID());
  }, []);

  const focusContent = useCallback(() => {
    if (cards.length === 0) return;

    const minX = Math.min(...cards.map((c) => c.x));
    const minY = Math.min(...cards.map((c) => c.y));
    const maxX = Math.max(...cards.map((c) => c.x + c.width));
    const maxY = Math.max(...cards.map((c) => c.y + CARD_HEIGHT_ESTIMATE));

    const padding = 120;
    const contentW = maxX - minX + padding * 2;
    const contentH = maxY - minY + padding * 2;

    const vw = window.innerWidth;
    const vh = window.innerHeight;

    const zoom = Math.min(
      Math.max(MIN_ZOOM, Math.min(vw / contentW, vh / contentH)),
      MAX_ZOOM,
    );

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    setCamera({
      x: vw / 2 - centerX * zoom,
      y: vh / 2 - centerY * zoom,
      zoom,
    });
  }, [cards]);

  const submitPrompt = useCallback(async () => {
    const text = prompt.trim();
    if (!text || !sessionId || isSubmitting) return;

    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const centerCanvas = screenToCanvas(vw / 2, vh / 2, camera);

    const index = placementIndexRef.current;
    const col = index % 3;
    const row = Math.floor(index / 3);
    placementIndexRef.current += 1;

    const draftCard = makePendingCard(
      crypto.randomUUID(),
      mode,
      text,
      centerCanvas.x - CARD_WIDTH / 2 + (col - 1) * CARD_STACK_X,
      centerCanvas.y - 140 + row * CARD_STACK_Y,
    );

    setCards((prev) => [...prev, draftCard]);
    setPrompt('');
    setIsSubmitting(true);

    try {
      const res = await fetch(`${API_BASE}/response`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: text,
          create_video: mode === 'video',
          session_id: sessionId,
        }),
      });

      const data = (await res.json()) as ApiResponse;
      setCards((prev) =>
        prev.map((c) => (c.id === draftCard.id ? toResolvedCard(c, data) : c)),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Request failed.';
      setCards((prev) =>
        prev.map((c) =>
          c.id === draftCard.id
            ? toResolvedCard(c, { ok: false, error: message, warnings: [] })
            : c,
        ),
      );
    } finally {
      setIsSubmitting(false);
    }
  }, [isSubmitting, mode, prompt, sessionId, camera]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement | HTMLInputElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        // Prevent default newline behavior
        e.preventDefault();
        
        // Only submit if there's actual content
        if (prompt.trim()) {
          void submitPrompt();
        }
      }
    },
    [submitPrompt, prompt],
  );

  /* ── textarea auto-resize ─────────────────────────────────────────── */
  useEffect(() => {
    const resize = (el: HTMLTextAreaElement | null) => {
      if (!el) return;
      el.style.height = 'auto';
      el.style.height = `${el.scrollHeight}px`;
    };
    resize(compactTextareaRef.current);
    resize(expandedTextareaRef.current);
  }, [prompt, dockExpanded]);

  /* ── dot-grid style driven by camera ──────────────────────────────── */
  const gridStyle: React.CSSProperties = {
    backgroundSize: `${24 * camera.zoom}px ${24 * camera.zoom}px`,
    backgroundPosition: `${camera.x}px ${camera.y}px`,
  };

  /* ── render ───────────────────────────────────────────────────────── */
  return (
    <main className="canvas-shell" data-theme={theme}>
      {/* canvas surface with dot grid */}
      <div
        ref={canvasRef}
        className={`canvas-surface${spaceHeldRef.current ? ' panning' : ''}`}
        style={gridStyle}
        onPointerDown={handleCanvasPointerDown}
      />

      {/* card layer */}
      <CanvasOverlay cards={cards} camera={camera} onStartDrag={beginCardDrag} />

      {/* prompt dock */}
      <section className={`query-dock ${dockExpanded ? 'dock-expanded' : 'dock-compact'}`}>
        {/* ─ compact bar ─ */}
        {!dockExpanded && (
          <div className="dock-compact-bar">
            <button
              type="button"
              className="dock-toggle-btn"
              onClick={() => setDockExpanded(true)}
              aria-label="Expand dock"
            >
              <ChevronUpIcon />
            </button>

            <button
              type="button"
              className="dock-toggle-btn"
              onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
              aria-label="Toggle theme"
            >
              {theme === 'light' ? <MoonIcon /> : <SunIcon />}
            </button>

            <textarea
              ref={compactTextareaRef}
              className="dock-compact-input"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={mode === 'video' ? 'Describe a video…' : 'Ask a question…'}
              rows={1}
            />

            <div className="dock-compact-actions">
              <button
                type="button"
                className={`mode-chip ${mode === 'video' ? 'video' : ''}`}
                onClick={() => setMode((m) => (m === 'text' ? 'video' : 'text'))}
              >
                {mode === 'video' ? '🎬 Video' : '💬 Text'}
              </button>

              <button
                type="button"
                className="send-btn"
                disabled={isSubmitting || !prompt.trim()}
                onClick={() => void submitPrompt()}
                aria-label="Send"
              >
                <SendIcon />
              </button>
            </div>
          </div>
        )}

        {/* ─ expanded panel ─ */}
        {dockExpanded && (
          <div className="dock-expanded-inner">
            <div className="dock-header">
              <div className="dock-title-area">
                <p className="dock-kicker">Glyph Canvas</p>
                <h1>Create learning cards on an infinite canvas</h1>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  type="button"
                  className="dock-toggle-btn"
                  onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
                  aria-label="Toggle theme"
                >
                  {theme === 'light' ? <MoonIcon /> : <SunIcon />}
                </button>
                <button
                  type="button"
                  className="dock-toggle-btn"
                  onClick={() => setDockExpanded(false)}
                  aria-label="Collapse dock"
                >
                  <ChevronDownIcon />
                </button>
              </div>
            </div>

            {/* mode toggle */}
            <div className="mode-toggle" role="tablist" aria-label="Query mode">
              <button
                type="button"
                className={`mode-pill ${mode === 'text' ? 'active' : ''}`}
                onClick={() => setMode('text')}
              >
                💬 Text
              </button>
              <button
                type="button"
                className={`mode-pill ${mode === 'video' ? 'active' : ''}`}
                onClick={() => setMode('video')}
              >
                🎬 Video
              </button>
            </div>

            {/* prompt area */}
            <label className="prompt-panel">
              <span>Prompt</span>
              <textarea
                ref={expandedTextareaRef}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  mode === 'video'
                    ? 'Describe a concept to visualize as a video card…'
                    : 'Ask a question and create a text card on the canvas…'
                }
                rows={1}
              />
            </label>

            {/* session details */}
            <div className="session-details">
              <div className="session-detail-item">
                <span className="detail-label">Session</span>
                <code>{sessionId ? sessionId.slice(0, 8) + '…' : '—'}</code>
              </div>
              <div className="session-detail-item">
                <span className="detail-label">Cards</span>
                <code>{cards.length}</code>
              </div>
              <div className="session-detail-item">
                <span className="detail-label">Mode</span>
                <code>{mode}</code>
              </div>
              <div className="session-detail-item">
                <span className="detail-label">Zoom</span>
                <code>{Math.round(camera.zoom * 100)}%</code>
              </div>
            </div>

            {/* actions */}
            <div className="dock-actions">
              <div className="dock-button-row">
                <button type="button" className="ghost-button" onClick={resetSession}>
                  <PlusIcon /> New session
                </button>
                <button
                  type="button"
                  className="ghost-button"
                  disabled={cards.length === 0}
                  onClick={focusContent}
                >
                  <CrosshairIcon /> Focus
                </button>
              </div>
              <button
                type="button"
                className="primary-button"
                disabled={isSubmitting || !prompt.trim()}
                onClick={() => void submitPrompt()}
              >
                {isSubmitting ? (
                  <>
                    <span className="spinner" /> Sending…
                  </>
                ) : (
                  <>
                    <SendIcon /> Create card
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </section>

      {/* tiny hint */}
      <p className="canvas-hint">
        Scroll to pan · Ctrl+Scroll to zoom · Drag card headers to move
      </p>
    </main>
  );
}
