import type { ApiResponse, Camera, CanvasCard, Mode } from '@/app/components/canvas/types';

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? 'https://glyph-production.up.railway.app';
export const CARD_WIDTH = 380;
export const CARD_MIN_WIDTH = 240;
export const CARD_STACK_X = 440;
export const CARD_STACK_Y = 340;
export const CARD_HEIGHT_ESTIMATE = 440;
export const MIN_ZOOM = 0.15;
export const MAX_ZOOM = 3;
export const ZOOM_SENSITIVITY = 0.0015;

export function screenToCanvas(sx: number, sy: number, cam: Camera) {
  return {
    x: (sx - cam.x) / cam.zoom,
    y: (sy - cam.y) / cam.zoom,
  };
}

export function normalizeText(value: unknown): string {
  if (typeof value === 'string') return value.trim();
  if (value == null) return '';
  if (Array.isArray(value)) return value.map(normalizeText).filter(Boolean).join('\n');

  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2).trim();
    } catch {
      return String(value).trim();
    }
  }

  return String(value).trim();
}

export function clamp(value: number, min: number, max: number): number {
  const upperBound = Math.max(min, max);
  return Math.min(Math.max(value, min), upperBound);
}

export function getCardWidth(viewportWidth: number): number {
  return Math.min(CARD_WIDTH, Math.max(CARD_MIN_WIDTH, viewportWidth - 32));
}

function isSuccessfulResponse(data: ApiResponse): boolean {
  if (typeof data.ok === 'boolean') return data.ok;
  if (typeof data.success === 'boolean') return data.success;
  return !data.error;
}

function getErrorMessage(data: ApiResponse): string {
  return (
    normalizeText(data.error) ||
    normalizeText(data.message) ||
    normalizeText(data.msg) ||
    normalizeText(data.response) ||
    normalizeText(data.content) ||
    'Request failed.'
  );
}

export function makePendingCard(
  id: string,
  mode: Mode,
  prompt: string,
  x: number,
  y: number,
  width = CARD_WIDTH,
): CanvasCard {
  return {
    id,
    kind: 'pending',
    mode,
    prompt,
    x,
    y,
    width,
    content: '',
    research: '',
    response: '',
    route: mode,
    error: '',
    warnings: [],
    statusLabel: 'Running',
  };
}

export function toResolvedCard(card: CanvasCard, data: ApiResponse): CanvasCard {
  const content = normalizeText(data.content ?? data.response);
  const research = normalizeText(data.research);
  const response = normalizeText(data.response);
  const warnings = (data.warnings ?? []).map((warning) => warning.trim()).filter(Boolean);

  if (!isSuccessfulResponse(data)) {
    return {
      ...card,
      kind: 'error',
      content: '',
      research: '',
      response: '',
      route: data.route ?? card.route,
      error: getErrorMessage(data),
      warnings: [],
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
