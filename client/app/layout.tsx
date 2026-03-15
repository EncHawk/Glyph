import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Glyph Canvas',
  description: 'Infinite canvas client for Glyph multimodal runs.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
