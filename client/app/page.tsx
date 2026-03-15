'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import MarkdownIt from 'markdown-it';
import { Tldraw, type Editor } from 'tldraw';

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

type CameraState = {
  x: number;
  y: number;
  z: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://127.0.0.1:8080';
const CARD_WIDTH = 360;
const CARD_HEIGHT_ESTIMATE = 440;
const CARD_STACK_X = 420;
const CARD_STACK_Y = 320;
const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
});

function normalizeText(value: unknown): string {
  if (typeof value === 'string') {
    return value.trim();
  }

  if (value == null) {
    return '';
  }

  if (Array.isArray(value)) {
    return value.map((item) => normalizeText(item)).filter(Boolean).join('\n');
  }

  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2).trim();
    } catch {
      return String(value).trim();
    }
  }

  return String(value).trim();
}

function makePendingCard(id: string, mode: Mode, prompt: string, x: number, y: number): CanvasCard {
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
  const warnings = (data.warnings ?? []).map((warning) => warning.trim()).filter(Boolean);

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

function MarkdownBlock({ content }: { content: string }) {
  return (
    <div
      className="markdown-block"
      dangerouslySetInnerHTML={{ __html: markdown.render(content) }}
    />
  );
}

function CanvasOverlay({
  cards,
  camera,
  onStartDrag,
}: {
  cards: CanvasCard[];
  camera: CameraState;
  onStartDrag: (event: React.PointerEvent<HTMLButtonElement>, cardId: string) => void;
}) {
  return (
    <div className="canvas-card-layer">
      {cards.map((card) => {
        const screenX = (card.x + camera.x) * camera.z;
        const screenY = (card.y + camera.y) * camera.z;

        return (
          <article
            key={card.id}
            className={`canvas-card ${card.kind}`}
            style={{
              width: `${card.width}px`,
              transform: `translate(${screenX}px, ${screenY}px) scale(${camera.z})`,
            }}
          >
            <button
              type="button"
              className="card-handle"
              onPointerDown={(event) => onStartDrag(event, card.id)}
            >
              <div>
                <p className="card-mode">{card.mode === 'video' ? 'Video call' : 'Text call'}</p>
                <h2>{card.prompt}</h2>
              </div>
              <span className="card-status">{card.statusLabel}</span>
            </button>

            <div className="card-body">
              <div className="card-meta">
                <span>{card.route || card.mode}</span>
                {card.warnings.length > 0 ? <span>{card.warnings.length} warning(s)</span> : null}
              </div>

              {card.kind === 'pending' ? (
                <div className="pending-block">
                  <span className="spinner" />
                  <p>Waiting for the backend agent to return.</p>
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
                    Open generated MP4
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

export default function Home() {
  const [mode, setMode] = useState<Mode>('text');
  const [prompt, setPrompt] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [cards, setCards] = useState<CanvasCard[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [editorReady, setEditorReady] = useState(false);
  const [camera, setCamera] = useState<CameraState>({ x: 0, y: 0, z: 1 });

  const editorRef = useRef<Editor | null>(null);
  const placementIndexRef = useRef(0);
  const dragStateRef = useRef<DragState | null>(null);

  useEffect(() => {
    setSessionId(crypto.randomUUID());
  }, []);

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      const dragState = dragStateRef.current;
      const editor = editorRef.current;
      if (!dragState || !editor) {
        return;
      }

      const pagePoint = editor.screenToPage({ x: event.clientX, y: event.clientY });
      setCards((current) =>
        current.map((card) =>
          card.id === dragState.cardId
            ? {
                ...card,
                x: pagePoint.x - dragState.offsetX,
                y: pagePoint.y - dragState.offsetY,
              }
            : card,
        ),
      );
    };

    const handlePointerUp = () => {
      dragStateRef.current = null;
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, []);

  useEffect(() => {
    if (!editorReady) {
      return;
    }

    let frameId = 0;

    const syncCamera = () => {
      const editor = editorRef.current;
      if (editor) {
        const next = editor.getCamera();
        setCamera((current) =>
          current.x === next.x && current.y === next.y && current.z === next.z
            ? current
            : { x: next.x, y: next.y, z: next.z },
        );
      }

      frameId = window.requestAnimationFrame(syncCamera);
    };

    frameId = window.requestAnimationFrame(syncCamera);
    return () => window.cancelAnimationFrame(frameId);
  }, [editorReady]);

  const onMount = useCallback((editor: Editor) => {
    editorRef.current = editor;
    editor.setCurrentTool('hand');
    setCamera(editor.getCamera());
    setEditorReady(true);
  }, []);

  const beginCardDrag = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>, cardId: string) => {
      event.preventDefault();
      event.stopPropagation();

      const editor = editorRef.current;
      const targetCard = cards.find((card) => card.id === cardId);
      if (!editor || !targetCard) {
        return;
      }

      const pagePoint = editor.screenToPage({ x: event.clientX, y: event.clientY });
      dragStateRef.current = {
        cardId,
        offsetX: pagePoint.x - targetCard.x,
        offsetY: pagePoint.y - targetCard.y,
      };
    },
    [cards],
  );

  const resetSession = useCallback(() => {
    setSessionId(crypto.randomUUID());
  }, []);

  const focusContent = useCallback(() => {
    const editor = editorRef.current;
    if (!editor || cards.length === 0) {
      return;
    }

    const minX = Math.min(...cards.map((card) => card.x));
    const minY = Math.min(...cards.map((card) => card.y));
    const maxX = Math.max(...cards.map((card) => card.x + card.width));
    const maxY = Math.max(...cards.map((card) => card.y + CARD_HEIGHT_ESTIMATE));

    editor.zoomToBounds(
      {
        x: minX - 120,
        y: minY - 120,
        w: maxX - minX + 240,
        h: maxY - minY + 240,
      },
      { animation: { duration: 240 } },
    );
  }, [cards]);

  const submitPrompt = useCallback(async () => {
    const nextPrompt = prompt.trim();
    const editor = editorRef.current;
    if (!nextPrompt || !editor || !sessionId || isSubmitting) {
      return;
    }

    const center = editor.getViewportPageBounds().center;
    const index = placementIndexRef.current;
    const column = index % 3;
    const row = Math.floor(index / 3);
    placementIndexRef.current += 1;

    const draftCard = makePendingCard(
      crypto.randomUUID(),
      mode,
      nextPrompt,
      center.x - 180 + column * CARD_STACK_X,
      center.y - 140 + row * CARD_STACK_Y,
    );

    setCards((current) => [...current, draftCard]);
    setPrompt('');
    setIsSubmitting(true);

    try {
      const response = await fetch(`${API_BASE}/response`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: nextPrompt,
          create_video: mode === 'video',
          session_id: sessionId,
        }),
      });

      const data = (await response.json()) as ApiResponse;
      setCards((current) =>
        current.map((card) => (card.id === draftCard.id ? toResolvedCard(card, data) : card)),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Request failed.';
      setCards((current) =>
        current.map((card) =>
          card.id === draftCard.id
            ? toResolvedCard(card, { ok: false, error: message, warnings: [] })
            : card,
        ),
      );
    } finally {
      setIsSubmitting(false);
    }
  }, [isSubmitting, mode, prompt, sessionId]);

  const handlePromptKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
        event.preventDefault();
        void submitPrompt();
      }
    },
    [submitPrompt],
  );

  return (
    <main className="canvas-shell">
      <div className="canvas-surface">
        <Tldraw hideUi onMount={onMount} />
      </div>

      <CanvasOverlay cards={cards} camera={camera} onStartDrag={beginCardDrag} />

      <section className="query-dock">
        <div className="dock-topline">
          <div>
            <p className="dock-kicker">Glyph Canvas</p>
            <h1>Infinite canvas with one custom card per server call.</h1>
          </div>
          <button type="button" className="ghost-button" onClick={resetSession}>
            New session
          </button>
        </div>

        <div className="mode-toggle" role="tablist" aria-label="Query mode">
          <button
            type="button"
            className={mode === 'text' ? 'mode-pill active' : 'mode-pill'}
            onClick={() => setMode('text')}
          >
            Text
          </button>
          <button
            type="button"
            className={mode === 'video' ? 'mode-pill active' : 'mode-pill'}
            onClick={() => setMode('video')}
          >
            Video
          </button>
        </div>

        <label className="prompt-panel">
          <span>Prompt</span>
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={handlePromptKeyDown}
            placeholder={
              mode === 'video'
                ? 'Explain a concept visually and create another video card on the canvas.'
                : 'Ask a question and create another text card on the canvas.'
            }
          />
        </label>

        <div className="dock-actions">
          <div className="session-readout">
            <span>Session</span>
            <code>{sessionId || 'booting...'}</code>
          </div>
          <div className="dock-button-row">
            <button
              type="button"
              className="bg-neutral-100 rounded-md p-1 cursor-pointer hover:bg-neutral-300 transition-all delay-75"
              disabled={!editorReady || cards.length === 0}
              onClick={focusContent}
            >
              Back to content
            </button>
            <button
              type="button"
              className="bg-blue-400 p-1 rounded-md text-white hover:bg-blue-600 transition-all delay-75 cursor-pointer"
              disabled={!editorReady || isSubmitting || !prompt.trim()}
              onClick={() => void submitPrompt()}
            >
              {isSubmitting ? 'Sending...' : 'Create card'}
            </button>
          </div>
        </div>

        <p className="dock-note">Drag the canvas to pan. Scroll to zoom. Drag card headers to reposition results.</p>
      </section>
    </main>
  );
}
