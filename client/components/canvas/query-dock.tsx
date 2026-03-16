'use client';

import {
  ChevronDownIcon,
  ChevronUpIcon,
  CrosshairIcon,
  MoonIcon,
  PlusIcon,
  SendIcon,
  SunIcon,
} from '@/components/canvas/icons';
import type { Camera, Mode } from '@/components/canvas/types';

export function QueryDock({
  cardsCount,
  camera,
  dockExpanded,
  isSubmitting,
  mode,
  prompt,
  sessionId,
  theme,
  compactTextareaRef,
  expandedTextareaRef,
  onFocusContent,
  onKeyDown,
  onModeChange,
  onPromptChange,
  onResetSession,
  onSend,
  onSetDockExpanded,
  onToggleTheme,
}: {
  cardsCount: number;
  camera: Camera;
  dockExpanded: boolean;
  isSubmitting: boolean;
  mode: Mode;
  prompt: string;
  sessionId: string;
  theme: 'light' | 'dark';
  compactTextareaRef: React.RefObject<HTMLTextAreaElement | null>;
  expandedTextareaRef: React.RefObject<HTMLTextAreaElement | null>;
  onFocusContent: () => void;
  onKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement | HTMLInputElement>) => void;
  onModeChange: (mode: Mode) => void;
  onPromptChange: (prompt: string) => void;
  onResetSession: () => void;
  onSend: () => void;
  onSetDockExpanded: (expanded: boolean) => void;
  onToggleTheme: () => void;
}) {
  return (
    <section className={`query-dock ${dockExpanded ? 'dock-expanded' : 'dock-compact'}`}>
      {!dockExpanded ? (
        <div className="dock-compact-bar">
          <button
            type="button"
            className="dock-toggle-btn"
            onClick={() => onSetDockExpanded(true)}
            aria-label="Expand dock"
          >
            <ChevronUpIcon />
          </button>

          <button
            type="button"
            className="dock-toggle-btn"
            onClick={onToggleTheme}
            aria-label="Toggle theme"
          >
            {theme === 'light' ? <MoonIcon /> : <SunIcon />}
          </button>

          <textarea
            ref={compactTextareaRef}
            className="dock-compact-input"
            value={prompt}
            onChange={(event) => onPromptChange(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder={mode === 'video' ? 'Describe a video...' : 'Ask a question...'}
            rows={1}
          />

          <div className="dock-compact-actions">
            <button
              type="button"
              className={`mode-chip ${mode === 'video' ? 'video' : ''}`}
              onClick={() => onModeChange(mode === 'text' ? 'video' : 'text')}
            >
              {mode === 'video' ? 'Video' : 'Text'}
            </button>

            <button
              type="button"
              className="send-btn"
              disabled={isSubmitting || !prompt.trim()}
              onClick={onSend}
              aria-label="Send"
            >
              <SendIcon />
            </button>
          </div>
        </div>
      ) : (
        <div className="dock-expanded-inner">
          <div className="dock-header">
            <div className="dock-title-area">
              <p className="dock-kicker">Glyph Canvas</p>
              <h1>Create learning cards on an infinite canvas</h1>
            </div>
            <div className="dock-header-actions">
              <button
                type="button"
                className="dock-toggle-btn"
                onClick={onToggleTheme}
                aria-label="Toggle theme"
              >
                {theme === 'light' ? <MoonIcon /> : <SunIcon />}
              </button>
              <button
                type="button"
                className="dock-toggle-btn"
                onClick={() => onSetDockExpanded(false)}
                aria-label="Collapse dock"
              >
                <ChevronDownIcon />
              </button>
            </div>
          </div>

          <div className="mode-toggle" role="tablist" aria-label="Query mode">
            <button
              type="button"
              className={`mode-pill ${mode === 'text' ? 'active' : ''}`}
              onClick={() => onModeChange('text')}
            >
              Text
            </button>
            <button
              type="button"
              className={`mode-pill ${mode === 'video' ? 'active' : ''}`}
              onClick={() => onModeChange('video')}
            >
              Video
            </button>
          </div>

          <label className="prompt-panel">
            <span>Prompt</span>
            <textarea
              ref={expandedTextareaRef}
              value={prompt}
              onChange={(event) => onPromptChange(event.target.value)}
              onKeyDown={onKeyDown}
              placeholder={
                mode === 'video'
                  ? 'Describe a concept to visualize as a video card...'
                  : 'Ask a question and create a text card on the canvas...'
              }
              rows={1}
            />
          </label>

          <div className="session-details">
            <div className="session-detail-item">
              <span className="detail-label">Session</span>
              <code>{sessionId ? `${sessionId.slice(0, 8)}...` : '-'}</code>
            </div>
            <div className="session-detail-item">
              <span className="detail-label">Cards</span>
              <code>{cardsCount}</code>
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

          <div className="dock-actions">
            <div className="dock-button-row">
              <button type="button" className="ghost-button" onClick={onResetSession}>
                <PlusIcon /> New session
              </button>
              <button
                type="button"
                className="ghost-button"
                disabled={cardsCount === 0}
                onClick={onFocusContent}
              >
                <CrosshairIcon /> Focus
              </button>
            </div>
            <button
              type="button"
              className="primary-button"
              disabled={isSubmitting || !prompt.trim()}
              onClick={onSend}
            >
              {isSubmitting ? (
                <>
                  <span className="spinner" /> Sending...
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
  );
}
