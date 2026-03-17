export type Mode = 'text' | 'video';
export type CardKind = 'pending' | 'text' | 'video' | 'error';

export type ApiResponse = {
  ok?: boolean;
  success?: boolean;
  route?: string | null;
  response?: string | null;
  content?: unknown;
  research?: unknown;
  error?: string | null;
  message?: unknown;
  msg?: unknown;
  warnings?: string[];
};

export type CanvasCard = {
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

export type DragState = {
  cardId: string;
  offsetX: number;
  offsetY: number;
};

export type Camera = {
  x: number;
  y: number;
  zoom: number;
};
