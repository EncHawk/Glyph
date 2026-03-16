'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Banner } from '@/app/components/banner';
import { CanvasOverlay } from '@/app/components/canvas/canvas-overlay';
import { QueryDock } from '@/app/components/canvas/query-dock';
import {
  API_BASE,
  CARD_HEIGHT_ESTIMATE,
  CARD_STACK_X,
  CARD_STACK_Y,
  CARD_WIDTH,
  MAX_ZOOM,
  MIN_ZOOM,
  ZOOM_SENSITIVITY,
  makePendingCard,
  screenToCanvas,
  toResolvedCard,
} from '@/app/components/canvas/helpers';
import type { ApiResponse, Camera, CanvasCard, DragState, Mode } from '@/app/components/canvas/types';

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

  useEffect(() => {
    setSessionId(crypto.randomUUID());
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !event.repeat) {
        const tag = (event.target as HTMLElement)?.tagName;
        if (tag === 'TEXTAREA' || tag === 'INPUT') return;
        event.preventDefault();
        spaceHeldRef.current = true;
      }
    };

    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === 'Space') {
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

  useEffect(() => {
    const element = canvasRef.current;
    if (!element) return;

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();

      if (event.ctrlKey || event.metaKey) {
        setCamera((current) => {
          const zoomFactor = 1 - event.deltaY * ZOOM_SENSITIVITY;
          const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current.zoom * zoomFactor));
          const ratio = newZoom / current.zoom;

          return {
            x: event.clientX - (event.clientX - current.x) * ratio,
            y: event.clientY - (event.clientY - current.y) * ratio,
            zoom: newZoom,
          };
        });
        return;
      }

      setCamera((current) => ({
        ...current,
        x: current.x - event.deltaX,
        y: current.y - event.deltaY,
      }));
    };

    element.addEventListener('wheel', onWheel, { passive: false });
    return () => element.removeEventListener('wheel', onWheel);
  }, []);

  useEffect(() => {
    const onPointerMove = (event: PointerEvent) => {
      const dragState = dragStateRef.current;
      if (dragState) {
        const canvasPoint = screenToCanvas(event.clientX, event.clientY, camera);
        setCards((current) =>
          current.map((card) =>
            card.id === dragState.cardId
              ? {
                  ...card,
                  x: canvasPoint.x - dragState.offsetX,
                  y: canvasPoint.y - dragState.offsetY,
                }
              : card,
          ),
        );
        return;
      }

      if (isPanningRef.current) {
        const dx = event.clientX - panStartRef.current.x;
        const dy = event.clientY - panStartRef.current.y;
        setCamera((current) => ({
          ...current,
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
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.button === 1 || (event.button === 0 && spaceHeldRef.current)) {
        event.preventDefault();
        isPanningRef.current = true;
        panStartRef.current = {
          x: event.clientX,
          y: event.clientY,
          camX: camera.x,
          camY: camera.y,
        };
      }
    },
    [camera],
  );

  const beginCardDrag = useCallback(
    (event: React.PointerEvent<HTMLButtonElement>, cardId: string) => {
      event.preventDefault();
      event.stopPropagation();

      const targetCard = cards.find((card) => card.id === cardId);
      if (!targetCard) return;

      const canvasPoint = screenToCanvas(event.clientX, event.clientY, camera);
      dragStateRef.current = {
        cardId,
        offsetX: canvasPoint.x - targetCard.x,
        offsetY: canvasPoint.y - targetCard.y,
      };
    },
    [cards, camera],
  );

  const resetSession = useCallback(() => {
    setSessionId(crypto.randomUUID());
  }, []);

  const focusContent = useCallback(() => {
    if (cards.length === 0) return;

    const minX = Math.min(...cards.map((card) => card.x));
    const minY = Math.min(...cards.map((card) => card.y));
    const maxX = Math.max(...cards.map((card) => card.x + card.width));
    const maxY = Math.max(...cards.map((card) => card.y + CARD_HEIGHT_ESTIMATE));

    const padding = 120;
    const contentWidth = maxX - minX + padding * 2;
    const contentHeight = maxY - minY + padding * 2;
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    const zoom = Math.min(
      Math.max(MIN_ZOOM, Math.min(viewportWidth / contentWidth, viewportHeight / contentHeight)),
      MAX_ZOOM,
    );

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    setCamera({
      x: viewportWidth / 2 - centerX * zoom,
      y: viewportHeight / 2 - centerY * zoom,
      zoom,
    });
  }, [cards]);

  const submitPrompt = useCallback(async () => {
    const text = prompt.trim();
    if (!text || !sessionId || isSubmitting) return;

    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const centerCanvas = screenToCanvas(viewportWidth / 2, viewportHeight / 2, camera);

    const index = placementIndexRef.current;
    const column = index % 3;
    const row = Math.floor(index / 3);
    placementIndexRef.current += 1;

    const draftCard = makePendingCard(
      crypto.randomUUID(),
      mode,
      text,
      centerCanvas.x - CARD_WIDTH / 2 + (column - 1) * CARD_STACK_X,
      centerCanvas.y - 140 + row * CARD_STACK_Y,
    );

    setCards((current) => [...current, draftCard]);
    setPrompt('');
    setIsSubmitting(true);

    try {
      const response = await fetch(`${API_BASE}/response`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: text,
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
  }, [camera, isSubmitting, mode, prompt, sessionId]);

  const handlePromptKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement | HTMLInputElement>) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        if (prompt.trim()) {
          void submitPrompt();
        }
      }
    },
    [prompt, submitPrompt],
  );

  useEffect(() => {
    const resize = (element: HTMLTextAreaElement | null) => {
      if (!element) return;
      element.style.height = 'auto';
      element.style.height = `${element.scrollHeight}px`;
    };

    resize(compactTextareaRef.current);
    resize(expandedTextareaRef.current);
  }, [prompt, dockExpanded]);

  const gridStyle: React.CSSProperties = {
    backgroundSize: `${24 * camera.zoom}px ${24 * camera.zoom}px`,
    backgroundPosition: `${camera.x}px ${camera.y}px`,
  };

  return (
    <main className="canvas-shell" data-theme={theme}>
      <Banner />

      <div
        ref={canvasRef}
        className={`canvas-surface${spaceHeldRef.current ? ' panning' : ''}`}
        style={gridStyle}
        onPointerDown={handleCanvasPointerDown}
      />

      <CanvasOverlay cards={cards} camera={camera} onStartDrag={beginCardDrag} />

      <QueryDock
        cardsCount={cards.length}
        camera={camera}
        compactTextareaRef={compactTextareaRef}
        dockExpanded={dockExpanded}
        expandedTextareaRef={expandedTextareaRef}
        isSubmitting={isSubmitting}
        mode={mode}
        prompt={prompt}
        sessionId={sessionId}
        theme={theme}
        onFocusContent={focusContent}
        onKeyDown={handlePromptKeyDown}
        onModeChange={setMode}
        onPromptChange={setPrompt}
        onResetSession={resetSession}
        onSend={() => void submitPrompt()}
        onSetDockExpanded={setDockExpanded}
        onToggleTheme={() => setTheme((current) => (current === 'light' ? 'dark' : 'light'))}
      />

      <p className="canvas-hint ">
        <span className="rounded-full px-1 bg-neutral-100 text-orange-600 mr-3">Beta</span>
        Scroll to pan · Ctrl+Scroll to zoom · Drag card headers to move
      </p>
    </main>
  );
}
